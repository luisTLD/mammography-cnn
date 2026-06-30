# =============================================================================
# Segmentacao e Classificacao de Imagens Mamograficas
# Processamento e Analise de Imagens
# Redes neurais sorteadas: VGG16 e EfficientNet-B0
# Dataset: RMLO (mama direita, incidencia medio-lateral obliqua)
# =============================================================================

import os
import re
import time
import glob
import queue
import threading

import numpy as np
import cv2

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights, EfficientNet_B0_Weights


RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_DATASET = os.path.join(RAIZ, "dataset")
DIR_MODELOS = os.path.join(RAIZ, "models")
DIR_SAIDAS = os.path.join(RAIZ, "outputs")
os.makedirs(DIR_MODELOS, exist_ok=True)
os.makedirs(DIR_SAIDAS, exist_ok=True)

TAMANHO_ENTRADA = 224
ANGULOS_AUMENTO = [-20, -10, 0, 10, 20]
MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSES_PASTA = ["D", "E", "F", "G"]
INDICE_CLASSE = {"D": 0, "E": 1, "F": 2, "G": 3}
NOME_BIRADS = {"D": "BI-RADS I", "E": "BI-RADS II", "F": "BI-RADS III", "G": "BI-RADS IV"}
ROTULOS_BINARIOS = ["I+II (baixa densidade)", "III+IV (alta densidade)"]
ROTULOS_QUATRO = ["I", "II", "III", "IV"]

LOTE = 16
EPOCAS_MAX = 40
PACIENCIA = 5
TAXA_APRENDIZADO = 1e-3
DECAIMENTO_PESO = 1e-4
FRACAO_VALIDACAO = 0.15
SEMENTE = 42
NUM_WORKERS = 0

PESOS_FIXOS = {2: [1.0, 1.0], 4: [1.0, 2.0, 2.0, 1.5]}
CONFIG_BATCH_RUN = [
    ("vgg", True, 25, 1e-3, 32, 0.5),
    ("vgg", False, 25, 1e-3, 32, 0.5),
    ("efficientnet", True, 50, 1e-3, 16, 0.3),
    ("efficientnet", False, 50, 1e-3, 16, 0.3),
]

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def carregar_imagem_cinza(caminho):
    bruta = cv2.imread(caminho, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if bruta is None:
        bruta = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if bruta is None:
        return None
    if bruta.ndim == 3:
        bruta = cv2.cvtColor(bruta, cv2.COLOR_BGR2GRAY)
    if bruta.dtype != np.uint8:
        bruta = cv2.normalize(bruta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return bruta


def segmentar_mama(imagem_original, lado_maximo=1024):
    altura0, largura0 = imagem_original.shape
    img = imagem_original.copy()
    img[:int(altura0 * 0.12), :int(largura0 * 0.20)] = 0
    escala = min(1.0, lado_maximo / max(altura0, largura0))
    if escala < 1.0:
        reduzida = cv2.resize(
            img,
            (int(largura0 * escala), int(altura0 * escala)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        reduzida = img

    altura, largura = reduzida.shape
    margem = max(5, int(min(altura, largura) * 0.02))
    reduzida[:margem, :] = 0
    reduzida[-margem:, :] = 0
    reduzida[:, :margem] = 0
    reduzida[:, -margem:] = 0

    suavizada = cv2.medianBlur(reduzida, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    realcada = clahe.apply(suavizada)
    limiar_otsu, binarizada = cv2.threshold(realcada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    aberta = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n_componentes, rotulada, estatisticas, _ = cv2.connectedComponentsWithStats(aberta)
    if n_componentes <= 1:
        vazia = np.zeros_like(imagem_original)
        return vazia, vazia

    maior = 1 + int(np.argmax(estatisticas[1:, cv2.CC_STAT_AREA]))
    mascara_mama = np.where(rotulada == maior, 255, 0).astype(np.uint8)

    limiar_baixo = max(10, int(limiar_otsu * 0.60))
    candidato = cv2.threshold(realcada, limiar_baixo, 255, cv2.THRESH_BINARY)[1]
    crescida = mascara_mama.copy()
    for _ in range(15):
        expandida = cv2.dilate(crescida, np.ones((9, 9), np.uint8))
        expandida = cv2.bitwise_and(expandida, candidato)
        if np.array_equal(expandida, crescida):
            break
        crescida = expandida

    crescida = cv2.morphologyEx(crescida, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n2, rot2, est2, _ = cv2.connectedComponentsWithStats(crescida)
    if n2 > 1:
        maior2 = 1 + int(np.argmax(est2[1:, cv2.CC_STAT_AREA]))
        mascara_mama = np.where(rot2 == maior2, 255, 0).astype(np.uint8)
    else:
        mascara_mama = crescida

    mascara_mama = cv2.morphologyEx(mascara_mama, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mascara_mama = cv2.dilate(mascara_mama, np.ones((7, 7), np.uint8), iterations=1)

    bm = max(8, int(min(altura, largura) * 0.06))
    mascara_mama[:bm, :] = 0
    mascara_mama[-bm:, :] = 0
    mascara_mama[:, :bm] = 0
    mascara_mama[:, -bm:] = 0

    padded = cv2.copyMakeBorder(mascara_mama, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, None, (0, 0), 128)
    padded_crop = padded[1:-1, 1:-1]
    buracos = (padded_crop == 0).astype(np.uint8) * 255
    mascara_mama = cv2.bitwise_or(mascara_mama, buracos)

    n3, rot3, est3, _ = cv2.connectedComponentsWithStats(mascara_mama)
    if n3 > 1:
        maior3 = 1 + int(np.argmax(est3[1:, cv2.CC_STAT_AREA]))
        mascara_mama = np.where(rot3 == maior3, 255, 0).astype(np.uint8)

    mascara = cv2.resize(mascara_mama, (largura0, altura0), interpolation=cv2.INTER_NEAREST)
    segmentada = cv2.bitwise_and(imagem_original, imagem_original, mask=mascara)
    return mascara, segmentada


def rotacionar(imagem, angulo):
    if angulo == 0:
        return imagem
    altura, largura = imagem.shape[:2]
    centro = (largura / 2.0, altura / 2.0)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(
        imagem,
        matriz,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def preparar_tensor(imagem_cinza, segmentar):
    if segmentar:
        _, imagem_cinza = segmentar_mama(imagem_cinza)
    redimensionada = cv2.resize(
        imagem_cinza, (TAMANHO_ENTRADA, TAMANHO_ENTRADA), interpolation=cv2.INTER_AREA
    )
    tres_canais = np.stack([redimensionada] * 3, axis=-1).astype(np.float32) / 255.0
    normalizada = (tres_canais - MEDIA_IMAGENET) / DESVIO_IMAGENET
    return torch.from_numpy(np.transpose(normalizada, (2, 0, 1))).float()


def numero_da_imagem(nome):
    encontrados = re.findall(r"\d+", os.path.basename(nome))
    if not encontrados:
        return None
    return int(encontrados[-1])


def eh_imagem_de_teste(nome):
    numero = numero_da_imagem(nome)
    if numero is None:
        return False
    return numero % 4 == 0


def classe_do_arquivo(caminho):
    pai = os.path.basename(os.path.dirname(caminho)).upper()
    if pai in INDICE_CLASSE:
        return pai
    inicial = os.path.basename(caminho)[:1].upper()
    if inicial in INDICE_CLASSE:
        return inicial
    return None


def listar_amostras(raiz):
    arquivos = []
    for extensao in ("*.png", "*.PNG", "*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        arquivos.extend(glob.glob(os.path.join(raiz, "**", extensao), recursive=True))
    treino, teste = [], []
    for caminho in sorted(set(arquivos)):
        letra = classe_do_arquivo(caminho)
        if letra is None:
            continue
        rotulo4 = INDICE_CLASSE[letra]
        registro = (caminho, rotulo4)
        if eh_imagem_de_teste(caminho):
            teste.append(registro)
        else:
            treino.append(registro)
    return treino, teste


def rotulo_para_modo(rotulo4, binario):
    if binario:
        return 0 if rotulo4 < 2 else 1
    return rotulo4


class DatasetMamografia(Dataset):
    def __init__(self, amostras, binario, usar_segmentacao, aumentar, cache):
        self.itens = []
        angulos = ANGULOS_AUMENTO if aumentar else [0]
        for caminho, rotulo4 in amostras:
            for angulo in angulos:
                self.itens.append((caminho, rotulo4, angulo))
        self.binario = binario
        self.usar_segmentacao = usar_segmentacao
        self.cache = cache

    def __len__(self):
        return len(self.itens)

    def _imagem_base(self, caminho):
        chave = (caminho, self.usar_segmentacao)
        if chave in self.cache:
            return self.cache[chave]
        imagem = carregar_imagem_cinza(caminho)
        if imagem is None:
            imagem = np.zeros((TAMANHO_ENTRADA, TAMANHO_ENTRADA), np.uint8)
        if self.usar_segmentacao:
            _, imagem = segmentar_mama(imagem)
        lado = min(imagem.shape)
        if lado > 512:
            fator = 512.0 / max(imagem.shape)
            imagem = cv2.resize(
                imagem,
                (int(imagem.shape[1] * fator), int(imagem.shape[0] * fator)),
                interpolation=cv2.INTER_AREA,
            )
        self.cache[chave] = imagem
        return imagem

    def __getitem__(self, indice):
        caminho, rotulo4, angulo = self.itens[indice]
        base = self._imagem_base(caminho)
        girada = rotacionar(base, angulo)
        redimensionada = cv2.resize(
            girada, (TAMANHO_ENTRADA, TAMANHO_ENTRADA), interpolation=cv2.INTER_AREA
        )
        tres = np.stack([redimensionada] * 3, axis=-1).astype(np.float32) / 255.0
        normalizada = (tres - MEDIA_IMAGENET) / DESVIO_IMAGENET
        tensor = torch.from_numpy(np.transpose(normalizada, (2, 0, 1))).float()
        return tensor, rotulo_para_modo(rotulo4, self.binario)


def separar_treino_validacao(amostras, fracao, semente):
    gerador = np.random.default_rng(semente)
    por_classe = {}
    for caminho, rotulo4 in amostras:
        por_classe.setdefault(rotulo4, []).append((caminho, rotulo4))
    treino, validacao = [], []
    for rotulo4, lista in por_classe.items():
        indices = gerador.permutation(len(lista))
        corte = max(1, int(len(lista) * fracao)) if len(lista) > 1 else 0
        for posicao, indice in enumerate(indices):
            if posicao < corte:
                validacao.append(lista[indice])
            else:
                treino.append(lista[indice])
    return treino, validacao


def pesos_de_classe(amostras, num_classes, binario):
    contagem = np.zeros(num_classes, dtype=np.float64)
    for _, rotulo4 in amostras:
        contagem[rotulo_para_modo(rotulo4, binario)] += 1
    contagem = np.maximum(contagem, 1.0)
    pesos = contagem.sum() / (num_classes * contagem)
    return torch.tensor(pesos, dtype=torch.float32)


def construir_modelo(nome, num_classes, dropout=0.5, congelar=False):
    if nome == "vgg":
        modelo = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if congelar:
            for parametro in modelo.features.parameters():
                parametro.requires_grad = False
        modelo.classifier[2] = nn.Dropout(p=dropout)
        modelo.classifier[5] = nn.Dropout(p=dropout)
        entradas = modelo.classifier[6].in_features
        modelo.classifier[6] = nn.Linear(entradas, num_classes)
    else:
        modelo = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        if congelar:
            for parametro in modelo.features.parameters():
                parametro.requires_grad = False
        entradas = modelo.classifier[1].in_features
        modelo.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(entradas, num_classes),
        )
    return modelo.to(DISPOSITIVO)


def ultima_camada_convolucional(modelo):
    alvo = None
    for modulo in modelo.modules():
        if isinstance(modulo, nn.Conv2d):
            alvo = modulo
    return alvo


def matriz_confusao(verdadeiros, preditos, num_classes):
    matriz = np.zeros((num_classes, num_classes), dtype=np.int64)
    for real, predito in zip(verdadeiros, preditos):
        matriz[real, predito] += 1
    return matriz


def metricas_binarias(matriz):
    vn, fp = matriz[0, 0], matriz[0, 1]
    fn, vp = matriz[1, 0], matriz[1, 1]
    eps = 1e-9
    sensibilidade = vp / (vp + fn + eps)
    especificidade = vn / (vn + fp + eps)
    precisao = vp / (vp + fp + eps)
    acuracia = (vp + vn) / (vp + vn + fp + fn + eps)
    f1 = 2 * precisao * sensibilidade / (precisao + sensibilidade + eps)
    return {
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "precisao": precisao,
        "acuracia": acuracia,
        "f1": f1,
    }


def metricas_multiclasse(matriz):
    eps = 1e-9
    total = matriz.sum()
    vp = np.diag(matriz).astype(np.float64)
    fp = matriz.sum(axis=0) - vp
    fn = matriz.sum(axis=1) - vp
    vn = total - (vp + fp + fn)
    sensibilidade = vp / (vp + fn + eps)
    especificidade = vn / (vn + fp + eps)
    acuracia = vp.sum() / (total + eps)
    return {
        "sensibilidade_por_classe": sensibilidade,
        "especificidade_por_classe": especificidade,
        "sensibilidade_media": sensibilidade.mean(),
        "especificidade_media": especificidade.mean(),
        "acuracia": acuracia,
    }


class GradCAM:
    def __init__(self, modelo, camada_alvo):
        self.modelo = modelo
        self.ativacoes = None
        self.gradientes = None
        camada_alvo.register_forward_hook(self._guardar_ativacoes)
        camada_alvo.register_full_backward_hook(self._guardar_gradientes)

    def _guardar_ativacoes(self, modulo, entrada, saida):
        self.ativacoes = saida.detach()

    def _guardar_gradientes(self, modulo, grad_entrada, grad_saida):
        self.gradientes = grad_saida[0].detach()

    def gerar(self, tensor_entrada, classe=None):
        self.modelo.eval()
        for modulo in self.modelo.modules():
            if hasattr(modulo, "inplace"):
                modulo.inplace = False
        tensor_entrada = tensor_entrada.unsqueeze(0).to(DISPOSITIVO)
        tensor_entrada.requires_grad_(True)
        with torch.enable_grad():
            saida = self.modelo(tensor_entrada)
            if classe is None:
                classe = int(saida.argmax(dim=1).item())
            self.modelo.zero_grad()
            saida[0, classe].backward()
        pesos = self.gradientes.mean(dim=(2, 3), keepdim=True)
        mapa = (pesos * self.ativacoes).sum(dim=1, keepdim=True)
        mapa = torch.relu(mapa)[0, 0].cpu().numpy()
        if mapa.max() > 0:
            mapa = mapa / mapa.max()
        mapa = cv2.resize(mapa, (TAMANHO_ENTRADA, TAMANHO_ENTRADA))
        return classe, mapa


def sobrepor_gradcam(imagem_cinza, mapa):
    altura, largura = imagem_cinza.shape[:2]
    base_rgb = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2RGB)
    mapa_red = cv2.resize(mapa, (largura, altura))
    cor = cv2.applyColorMap(np.uint8(255 * mapa_red), cv2.COLORMAP_JET)
    cor = cv2.cvtColor(cor, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 0.55, cor, 0.45, 0)


def treinar_modelo(modelo, carregador_treino, carregador_validacao, pesos, registrar, parar, epocas, lr=TAXA_APRENDIZADO, congelar=False):
    criterio = nn.CrossEntropyLoss(weight=pesos.to(DISPOSITIVO))
    if congelar:
        parametros = [p for p in modelo.parameters() if p.requires_grad]
        otimizador = optim.Adam(parametros, lr=lr, weight_decay=DECAIMENTO_PESO)
    else:
        otimizador = optim.Adam([
            {"params": modelo.features.parameters(), "lr": lr * 0.1},
            {"params": modelo.classifier.parameters(), "lr": lr},
        ])
    agendador = optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.5, patience=3)

    historico = {"perda_treino": [], "perda_validacao": [], "acuracia_validacao": []}
    melhor_perda = float("inf")
    melhores_pesos = None
    sem_melhora = 0

    for epoca in range(epocas):
        if parar.is_set():
            break
        modelo.train()
        perda_acumulada = 0.0
        for entradas, alvos in carregador_treino:
            if parar.is_set():
                break
            entradas = entradas.to(DISPOSITIVO)
            alvos = torch.as_tensor(alvos).to(DISPOSITIVO)
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = criterio(saidas, alvos)
            perda.backward()
            otimizador.step()
            perda_acumulada += perda.item() * entradas.size(0)
        perda_treino = perda_acumulada / max(1, len(carregador_treino.dataset))

        modelo.eval()
        perda_val = 0.0
        acertos = 0
        total = 0
        with torch.no_grad():
            for entradas, alvos in carregador_validacao:
                entradas = entradas.to(DISPOSITIVO)
                alvos = torch.as_tensor(alvos).to(DISPOSITIVO)
                saidas = modelo(entradas)
                perda_val += criterio(saidas, alvos).item() * entradas.size(0)
                preditos = saidas.argmax(dim=1)
                acertos += (preditos == alvos).sum().item()
                total += alvos.size(0)
        perda_val = perda_val / max(1, len(carregador_validacao.dataset))
        acuracia_val = acertos / max(1, total)

        historico["perda_treino"].append(perda_treino)
        historico["perda_validacao"].append(perda_val)
        historico["acuracia_validacao"].append(acuracia_val)
        agendador.step(perda_val)
        registrar(
            "Epoca %02d/%d | perda treino %.4f | perda val %.4f | acuracia val %.3f"
            % (epoca + 1, epocas, perda_treino, perda_val, acuracia_val)
        )

        if perda_val < melhor_perda - 1e-4:
            melhor_perda = perda_val
            melhores_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                registrar("Parada antecipada (sem melhora em %d epocas)." % PACIENCIA)
                break

    if melhores_pesos is not None:
        modelo.load_state_dict(melhores_pesos)
    return historico


def avaliar_modelo(modelo, carregador):
    modelo.eval()
    verdadeiros, preditos = [], []
    inicio = time.time()
    with torch.no_grad():
        for entradas, alvos in carregador:
            entradas = entradas.to(DISPOSITIVO)
            saidas = modelo(entradas)
            lote_preditos = saidas.argmax(dim=1).cpu().numpy()
            preditos.extend(lote_preditos.tolist())
            verdadeiros.extend(list(np.asarray(alvos)))
    tempo = time.time() - inicio
    return [int(v) for v in verdadeiros], [int(p) for p in preditos], tempo


def salvar_figura_convergencia(historico, caminho, titulo):
    figura = Figure(figsize=(7, 5), dpi=100)
    FigureCanvasAgg(figura)
    eixo = figura.add_subplot(111)
    eixo.plot(historico["perda_treino"], label="perda treino")
    eixo.plot(historico["perda_validacao"], label="perda validacao")
    eixo.plot(historico["acuracia_validacao"], label="acuracia validacao")
    eixo.set_xlabel("epoca")
    eixo.set_title(titulo)
    eixo.legend()
    eixo.grid(True, alpha=0.3)
    figura.savefig(caminho, bbox_inches="tight")


def salvar_figura_matriz(matriz, rotulos, caminho, titulo):
    figura = Figure(figsize=(6, 5), dpi=100)
    FigureCanvasAgg(figura)
    eixo = figura.add_subplot(111)
    imagem = eixo.imshow(matriz, cmap="Blues")
    eixo.set_xticks(range(len(rotulos)))
    eixo.set_yticks(range(len(rotulos)))
    eixo.set_xticklabels(rotulos, rotation=30, ha="right")
    eixo.set_yticklabels(rotulos)
    eixo.set_xlabel("Predito")
    eixo.set_ylabel("Verdadeiro")
    eixo.set_title(titulo)
    limite = matriz.max() / 2.0 if matriz.max() > 0 else 0.5
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            eixo.text(
                j, i, str(matriz[i, j]), ha="center", va="center",
                color="white" if matriz[i, j] > limite else "black",
            )
    figura.colorbar(imagem, ax=eixo, fraction=0.046)
    figura.savefig(caminho, bbox_inches="tight")


def escrever_relatorio_txt(caminho, hiper, tempo_treino, tempo_inf, matriz, binario, historico):
    linhas = []
    linhas.append("Rede: %s | Tarefa: %s" % (hiper["rede"], hiper["tarefa"]))
    linhas.append("Congelamento: %s | Peso: %s" % (hiper.get("congelamento"), hiper.get("peso")))
    linhas.append("Imagens segmentadas: %s" % hiper["segmentacao"])
    linhas.append(
        "Hiperparametros: epocas_max=%d, epocas_rodadas=%d, lr=%s, batch=%d, dropout=%.2f"
        % (hiper["epocas_max"], hiper["epocas_rodadas"], hiper["lr"], hiper["batch"], hiper["dropout"])
    )
    linhas.append("Tempo de treino: %.1f s" % tempo_treino)
    linhas.append("Tempo de inferencia no teste: %.3f s" % tempo_inf)
    linhas.append("")
    linhas.append("Matriz de confusao (linha = verdadeiro, coluna = predito):")
    linhas.append(str(matriz))
    linhas.append("")
    if binario:
        m = metricas_binarias(matriz)
        linhas.append("Sensibilidade: %.4f" % m["sensibilidade"])
        linhas.append("Especificidade: %.4f" % m["especificidade"])
        linhas.append("Precisao: %.4f" % m["precisao"])
        linhas.append("Acuracia: %.4f" % m["acuracia"])
        linhas.append("Escore F1: %.4f" % m["f1"])
    else:
        m = metricas_multiclasse(matriz)
        linhas.append("Acuracia: %.4f" % m["acuracia"])
        linhas.append("Sensibilidade media: %.4f" % m["sensibilidade_media"])
        linhas.append("Especificidade media: %.4f" % m["especificidade_media"])
        linhas.append("Sensibilidade por classe: %s" % np.round(m["sensibilidade_por_classe"], 4).tolist())
        linhas.append("Especificidade por classe: %s" % np.round(m["especificidade_por_classe"], 4).tolist())
    linhas.append("")
    linhas.append("Historico por epoca (perda_treino, perda_validacao, acuracia_validacao):")
    for i in range(len(historico["perda_treino"])):
        linhas.append(
            "epoca %02d: %.4f, %.4f, %.4f"
            % (i + 1, historico["perda_treino"][i], historico["perda_validacao"][i], historico["acuracia_validacao"][i])
        )
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))


class Aplicacao(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segmentacao e Classificacao de Mamografias - VGG16 / EfficientNet-B0")
        self.geometry("1280x820")

        self.imagem_atual = None
        self.caminho_imagem = None
        self.mascara_atual = None
        self.segmentada_atual = None
        self.raiz_dataset = DIR_DATASET
        self.amostras_treino = []
        self.amostras_teste = []
        self.cache_imagens = {}
        self.modelos = {}
        self.fila = queue.Queue()
        self.parar_treino = threading.Event()

        torch.manual_seed(SEMENTE)
        np.random.seed(SEMENTE)

        self._montar_interface()
        self.after(120, self._processar_fila)

    def _montar_interface(self):
        painel = ttk.Frame(self, padding=8)
        painel.pack(side=tk.LEFT, fill=tk.Y)

        area = ttk.Frame(self, padding=4)
        area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figura = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figura, master=area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        barra = NavigationToolbar2Tk(self.canvas, area)
        barra.update()

        self.texto_log = tk.Text(area, height=9, wrap="word")
        self.texto_log.pack(fill=tk.X, side=tk.BOTTOM)

        def secao(titulo):
            ttk.Label(painel, text=titulo, font=("Segoe UI", 9, "bold")).pack(
                anchor="w", pady=(10, 2)
            )

        def botao(texto, comando):
            ttk.Button(painel, text=texto, command=comando, width=30).pack(pady=1)

        secao("Imagem (itens a, b, c)")
        botao("Abrir imagem (PNG/TIFF)", self.acao_abrir_imagem)
        botao("Segmentar mama", self.acao_segmentar)

        secao("Aumento de dados (item e)")
        botao("Visualizar aumento (rotacoes)", self.acao_aumento)

        secao("Dataset (item d)")
        botao("Selecionar diretorio do dataset", self.acao_selecionar_dataset)
        botao("Resumo treino/teste", self.acao_resumo_dataset)

        secao("Treino e avaliacao (item f)")
        ttk.Label(painel, text="Rede:").pack(anchor="w")
        self.var_rede = tk.StringVar(value="vgg")
        ttk.Combobox(
            painel, textvariable=self.var_rede, values=["vgg", "efficientnet"],
            state="readonly", width=27,
        ).pack()
        ttk.Label(painel, text="Tarefa:").pack(anchor="w")
        self.var_modo = tk.StringVar(value="binaria")
        ttk.Combobox(
            painel, textvariable=self.var_modo, values=["binaria", "quatro"],
            state="readonly", width=27,
        ).pack()
        ttk.Label(painel, text="Congelamento:").pack(anchor="w")
        self.var_congelar = tk.StringVar(value="sem congelamento")
        ttk.Combobox(
            painel, textvariable=self.var_congelar,
            values=["sem congelamento", "com congelamento"],
            state="readonly", width=27,
        ).pack()
        self.var_segmentar = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            painel, text="Usar imagens segmentadas", variable=self.var_segmentar
        ).pack(anchor="w")
        ttk.Label(painel, text="Epocas:").pack(anchor="w")
        self.var_epocas = tk.IntVar(value=EPOCAS_MAX)
        ttk.Spinbox(painel, from_=1, to=200, textvariable=self.var_epocas, width=27).pack()
        ttk.Label(painel, text="Taxa de aprendizado (LR):").pack(anchor="w")
        self.var_lr = tk.StringVar(value=str(TAXA_APRENDIZADO))
        ttk.Combobox(
            painel, textvariable=self.var_lr,
            values=["0.01", "0.005", "0.001", "0.0005", "0.0001"], width=27,
        ).pack()
        ttk.Label(painel, text="Batch size:").pack(anchor="w")
        self.var_lote = tk.IntVar(value=LOTE)
        ttk.Spinbox(painel, from_=1, to=128, textvariable=self.var_lote, width=27).pack()
        ttk.Label(painel, text="Dropout:").pack(anchor="w")
        self.var_dropout = tk.DoubleVar(value=0.5)
        ttk.Spinbox(
            painel, from_=0.0, to=0.9, increment=0.1,
            textvariable=self.var_dropout, width=27,
        ).pack()
        ttk.Label(painel, text="Peso da perda:").pack(anchor="w")
        self.var_peso = tk.StringVar(value="automatico")
        ttk.Combobox(
            painel, textvariable=self.var_peso, values=["automatico", "fixo"],
            state="readonly", width=27,
        ).pack()
        botao("Treinar rede", self.acao_treinar)
        botao("Rodar os 4 experimentos (batch)", self.acao_rodar_tudo)
        botao("Interromper treino", self.acao_parar)
        botao("Avaliar conjunto de teste", self.acao_avaliar)
        botao("Carregar pesos salvos", self.acao_carregar_pesos)

        secao("Explicabilidade (item g)")
        botao("Classificar imagem + Grad-CAM", self.acao_gradcam)

        ttk.Label(
            painel,
            text="Batch-run (4 experimentos):\nVGG: 25 ep | batch 32 | dropout 0.5\nEfficientNet: 50 ep | batch 16 | dropout 0.3\nlr 1e-3 | usa o congelamento escolhido acima",
            justify="left",
        ).pack(anchor="w", pady=(12, 0))
        ttk.Label(painel, text="Dispositivo: %s" % DISPOSITIVO).pack(anchor="w", pady=(6, 0))

    def registrar(self, mensagem):
        self.fila.put(("log", mensagem))

    def _processar_fila(self):
        try:
            while True:
                tipo, conteudo = self.fila.get_nowait()
                if tipo == "log":
                    self.texto_log.insert(tk.END, conteudo + "\n")
                    self.texto_log.see(tk.END)
                elif tipo == "figura":
                    conteudo()
        except queue.Empty:
            pass
        self.after(120, self._processar_fila)

    def _desenhar(self, funcao):
        self.fila.put(("figura", funcao))

    def acao_abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        imagem = carregar_imagem_cinza(caminho)
        if imagem is None:
            messagebox.showerror("Erro", "Nao foi possivel ler a imagem.")
            return
        self.imagem_atual = imagem
        self.caminho_imagem = caminho
        self.mascara_atual = None
        self.segmentada_atual = None
        self.figura.clear()
        eixo = self.figura.add_subplot(111)
        eixo.imshow(imagem, cmap="gray")
        eixo.set_title("%s  %dx%d" % (os.path.basename(caminho), imagem.shape[1], imagem.shape[0]))
        eixo.axis("off")
        self.canvas.draw()
        self.registrar("Imagem aberta: %s" % os.path.basename(caminho))

    def acao_segmentar(self):
        if self.imagem_atual is None:
            messagebox.showinfo("Aviso", "Abra uma imagem primeiro.")
            return
        mascara, segmentada = segmentar_mama(self.imagem_atual)
        self.mascara_atual = mascara
        self.segmentada_atual = segmentada
        cobertura = 100.0 * (mascara > 0).sum() / mascara.size
        self.figura.clear()
        titulos = ["Original", "Mascara", "Segmentada"]
        imagens = [self.imagem_atual, mascara, segmentada]
        for i, (img, titulo) in enumerate(zip(imagens, titulos), 1):
            eixo = self.figura.add_subplot(1, 3, i)
            eixo.imshow(img, cmap="gray")
            eixo.set_title(titulo)
            eixo.axis("off")
        self.canvas.draw()
        self.registrar("Segmentacao concluida. Area da mama: %.1f%%" % cobertura)

    def acao_aumento(self):
        if self.imagem_atual is None:
            messagebox.showinfo("Aviso", "Abra uma imagem primeiro.")
            return
        base = self.segmentada_atual if self.segmentada_atual is not None else self.imagem_atual
        self.figura.clear()
        for i, angulo in enumerate(ANGULOS_AUMENTO, 1):
            eixo = self.figura.add_subplot(1, len(ANGULOS_AUMENTO), i)
            eixo.imshow(rotacionar(base, angulo), cmap="gray")
            eixo.set_title("%d graus" % angulo)
            eixo.axis("off")
        self.canvas.draw()
        self.registrar("Aumento de dados: 5 rotacoes geradas (-20 a 20 graus).")

    def acao_selecionar_dataset(self):
        diretorio = filedialog.askdirectory(initialdir=self.raiz_dataset)
        if not diretorio:
            return
        self.raiz_dataset = diretorio
        self.cache_imagens = {}
        self.acao_resumo_dataset()

    def acao_resumo_dataset(self):
        self.amostras_treino, self.amostras_teste = listar_amostras(self.raiz_dataset)
        self.registrar("Diretorio: %s" % self.raiz_dataset)
        for nome, amostras in [("TREINO", self.amostras_treino), ("TESTE", self.amostras_teste)]:
            contagem = {c: 0 for c in CLASSES_PASTA}
            for caminho, rotulo4 in amostras:
                contagem[CLASSES_PASTA[rotulo4]] += 1
            self.registrar(
                "%s: %d imagens  (D=%d E=%d F=%d G=%d)"
                % (nome, len(amostras), contagem["D"], contagem["E"], contagem["F"], contagem["G"])
            )
        self.registrar(
            "Treino apos aumento (x5 rotacoes): %d imagens." % (len(self.amostras_treino) * 5)
        )

    def _construir_carregadores(self, binario, usar_segmentacao, lote):
        treino, validacao = separar_treino_validacao(
            self.amostras_treino, FRACAO_VALIDACAO, SEMENTE
        )
        ds_treino = DatasetMamografia(treino, binario, usar_segmentacao, True, self.cache_imagens)
        ds_validacao = DatasetMamografia(validacao, binario, usar_segmentacao, False, self.cache_imagens)
        ds_teste = DatasetMamografia(self.amostras_teste, binario, usar_segmentacao, False, self.cache_imagens)
        carregador_treino = DataLoader(ds_treino, batch_size=lote, shuffle=True, num_workers=NUM_WORKERS)
        carregador_validacao = DataLoader(ds_validacao, batch_size=lote, shuffle=False, num_workers=NUM_WORKERS)
        carregador_teste = DataLoader(ds_teste, batch_size=lote, shuffle=False, num_workers=NUM_WORKERS)
        return treino, carregador_treino, carregador_validacao, carregador_teste

    def acao_parar(self):
        self.parar_treino.set()
        self.registrar("Solicitada interrupcao do treino...")

    def acao_treinar(self):
        if not self.amostras_treino:
            messagebox.showinfo("Aviso", "Selecione o diretorio do dataset primeiro.")
            return
        rede = self.var_rede.get()
        binario = self.var_modo.get() == "binaria"
        usar_segmentacao = bool(self.var_segmentar.get())
        epocas = int(self.var_epocas.get())
        try:
            lr = float(self.var_lr.get())
        except ValueError:
            lr = TAXA_APRENDIZADO
        lote = max(1, int(self.var_lote.get()))
        dropout = float(self.var_dropout.get())
        modo_peso = self.var_peso.get()
        congelar = self._congelar_selecionado()
        self.parar_treino.clear()
        config = (rede, binario, usar_segmentacao, epocas, lr, lote, dropout, modo_peso, congelar)
        threading.Thread(target=lambda: self._rodar_experimento(*config), daemon=True).start()

    def _rodar_experimento(self, rede, binario, usar_segmentacao, epocas, lr, lote, dropout, modo_peso, congelar):
        inicio = time.time()
        num_classes = 2 if binario else 4
        nome_tarefa = "binaria" if binario else "4 classes"
        self.registrar(
            "== Treino: %s | %s | congelamento=%s | peso=%s | seg=%s | epocas<=%d | lr=%s | batch=%d | dropout=%.2f"
            % (rede, nome_tarefa, congelar, modo_peso, usar_segmentacao, epocas, lr, lote, dropout)
        )
        treino, c_treino, c_val, c_teste = self._construir_carregadores(binario, usar_segmentacao, lote)
        if modo_peso == "fixo":
            pesos = torch.tensor(PESOS_FIXOS[num_classes], dtype=torch.float32)
        else:
            pesos = pesos_de_classe(treino, num_classes, binario)
        modelo = construir_modelo(rede, num_classes, dropout, congelar)
        historico = treinar_modelo(
            modelo, c_treino, c_val, pesos, self.registrar, self.parar_treino, epocas, lr, congelar
        )
        tempo_treino = time.time() - inicio
        chave = "%s_%s_%s_%s" % (rede, "bin" if binario else "quatro", "cong" if congelar else "sem", "pesofix" if modo_peso == "fixo" else "pesoauto")
        caminho_pesos = os.path.join(DIR_MODELOS, "%s.pth" % chave)
        torch.save(modelo.state_dict(), caminho_pesos)
        self.modelos[chave] = (modelo, usar_segmentacao)
        verdadeiros, preditos, tempo_inf = avaliar_modelo(modelo, c_teste)
        matriz = matriz_confusao(verdadeiros, preditos, num_classes)
        rotulos = ROTULOS_BINARIOS if binario else ROTULOS_QUATRO
        titulo = "%s (%s)" % (rede, nome_tarefa)
        prefixo = os.path.join(DIR_SAIDAS, chave)
        salvar_figura_convergencia(historico, prefixo + "_convergencia.png", "Convergencia - " + titulo)
        salvar_figura_matriz(matriz, rotulos, prefixo + "_matriz_confusao.png", "Matriz de confusao - " + titulo)
        hiper = {
            "rede": rede, "tarefa": nome_tarefa, "segmentacao": usar_segmentacao,
            "congelamento": congelar, "peso": modo_peso,
            "epocas_max": epocas, "epocas_rodadas": len(historico["perda_treino"]),
            "lr": lr, "batch": lote, "dropout": dropout,
        }
        escrever_relatorio_txt(prefixo + "_metricas.txt", hiper, tempo_treino, tempo_inf, matriz, binario, historico)
        self.registrar("Pesos salvos em %s" % caminho_pesos)
        self.registrar(
            "Salvo em outputs/: %s_convergencia.png | %s_matriz_confusao.png | %s_metricas.txt"
            % (chave, chave, chave)
        )
        self.registrar("Tempo de treino: %.1f s | inferencia (teste): %.3f s" % (tempo_treino, tempo_inf))
        self._desenhar(lambda: self._plotar_convergencia(historico, rede, binario))
        self._mostrar_metricas(verdadeiros, preditos, binario, tempo_inf, rede)

    def acao_rodar_tudo(self):
        if not self.amostras_treino:
            messagebox.showinfo("Aviso", "Selecione o diretorio do dataset primeiro.")
            return
        usar_segmentacao = bool(self.var_segmentar.get())
        modo_peso = self.var_peso.get()
        congelar = self._congelar_selecionado()
        self.parar_treino.clear()
        experimentos = [
            (rede, binario, usar_segmentacao, epocas, lr, lote, dropout, modo_peso, congelar)
            for (rede, binario, epocas, lr, lote, dropout) in CONFIG_BATCH_RUN
        ]

        def trabalho():
            inicio_total = time.time()
            self.registrar("=== Iniciando os 4 experimentos (congelamento=%s, peso=%s) ===" % (congelar, modo_peso))
            for indice, config in enumerate(experimentos, 1):
                if self.parar_treino.is_set():
                    self.registrar("Execucao interrompida pelo usuario.")
                    break
                self.registrar(">>> Experimento %d de 4" % indice)
                self._rodar_experimento(*config)
            self.registrar("=== Concluido. Tempo total: %.1f min ===" % ((time.time() - inicio_total) / 60.0))
            self.registrar("Tudo salvo em models/ (pesos .pth) e outputs/ (curvas, matrizes e metricas).")

        threading.Thread(target=trabalho, daemon=True).start()

    def _congelar_selecionado(self):
        return self.var_congelar.get() == "com congelamento"

    def _chave(self, rede, binario):
        pesotag = "pesofix" if self.var_peso.get() == "fixo" else "pesoauto"
        congtag = "cong" if self._congelar_selecionado() else "sem"
        return "%s_%s_%s_%s" % (rede, "bin" if binario else "quatro", congtag, pesotag)

    def acao_avaliar(self):
        rede = self.var_rede.get()
        binario = self.var_modo.get() == "binaria"
        chave = self._chave(rede, binario)
        if chave not in self.modelos:
            messagebox.showinfo("Aviso", "Treine ou carregue os pesos desta configuracao primeiro.")
            return
        if not self.amostras_teste:
            messagebox.showinfo("Aviso", "Selecione o diretorio do dataset primeiro.")
            return
        modelo, usar_segmentacao = self.modelos[chave]

        def trabalho():
            ds_teste = DatasetMamografia(
                self.amostras_teste, binario, usar_segmentacao, False, self.cache_imagens
            )
            carregador = DataLoader(ds_teste, batch_size=LOTE, shuffle=False, num_workers=NUM_WORKERS)
            verdadeiros, preditos, tempo_inf = avaliar_modelo(modelo, carregador)
            self._mostrar_metricas(verdadeiros, preditos, binario, tempo_inf, rede)

        threading.Thread(target=trabalho, daemon=True).start()

    def acao_carregar_pesos(self):
        caminho = filedialog.askopenfilename(initialdir=DIR_MODELOS, filetypes=[("Pesos", "*.pth")])
        if not caminho:
            return
        # Infere a configuracao a partir do nome do arquivo (ex.: vgg_quatro_cong_pesofix.pth)
        # e reflete na interface, para a chave de avaliacao/Grad-CAM bater com os pesos carregados.
        partes = os.path.splitext(os.path.basename(caminho))[0].lower().split("_")
        if "efficientnet" in partes:
            rede = "efficientnet"
        elif "vgg" in partes:
            rede = "vgg"
        else:
            rede = self.var_rede.get()
        if "bin" in partes:
            binario = True
        elif "quatro" in partes:
            binario = False
        else:
            binario = self.var_modo.get() == "binaria"
        if "cong" in partes:
            congelar = True
        elif "sem" in partes:
            congelar = False
        else:
            congelar = self._congelar_selecionado()
        if "pesofix" in partes:
            modo_peso = "fixo"
        elif "pesoauto" in partes:
            modo_peso = "automatico"
        else:
            modo_peso = self.var_peso.get()

        self.var_rede.set(rede)
        self.var_modo.set("binaria" if binario else "quatro")
        self.var_congelar.set("com congelamento" if congelar else "sem congelamento")
        self.var_peso.set(modo_peso)

        num_classes = 2 if binario else 4
        # congelar=False na construcao para inferencia/Grad-CAM (a arquitetura e identica;
        # o congelamento so afeta o treino).
        modelo = construir_modelo(rede, num_classes, float(self.var_dropout.get()), congelar=False)
        modelo.load_state_dict(torch.load(caminho, map_location=DISPOSITIVO))
        modelo.eval()
        chave = self._chave(rede, binario)
        self.modelos[chave] = (modelo, bool(self.var_segmentar.get()))
        self.registrar(
            "Pesos carregados: %s (rede=%s | tarefa=%s | congelamento=%s | peso=%s)"
            % (os.path.basename(caminho), rede, "binaria" if binario else "4 classes", congelar, modo_peso)
        )

    def _plotar_convergencia(self, historico, rede, binario):
        self.figura.clear()
        eixo = self.figura.add_subplot(111)
        eixo.plot(historico["perda_treino"], label="perda treino")
        eixo.plot(historico["perda_validacao"], label="perda validacao")
        eixo.plot(historico["acuracia_validacao"], label="acuracia validacao")
        eixo.set_xlabel("epoca")
        eixo.set_title("Convergencia - %s (%s)" % (rede, "binaria" if binario else "4 classes"))
        eixo.legend()
        eixo.grid(True, alpha=0.3)
        self.canvas.draw()

    def _mostrar_metricas(self, verdadeiros, preditos, binario, tempo_inf, rede):
        if binario:
            matriz = matriz_confusao(verdadeiros, preditos, 2)
            metricas = metricas_binarias(matriz)
            self.registrar("=== Metricas binarias (%s) ===" % rede)
            self.registrar("Tempo de classificacao do teste: %.3f s" % tempo_inf)
            self.registrar("Sensibilidade: %.4f" % metricas["sensibilidade"])
            self.registrar("Especificidade: %.4f" % metricas["especificidade"])
            self.registrar("Precisao: %.4f" % metricas["precisao"])
            self.registrar("Acuracia: %.4f" % metricas["acuracia"])
            self.registrar("Escore F1: %.4f" % metricas["f1"])
            self._desenhar(lambda: self._plotar_matriz(matriz, ROTULOS_BINARIOS, "Binaria - %s" % rede))
        else:
            matriz = matriz_confusao(verdadeiros, preditos, 4)
            metricas = metricas_multiclasse(matriz)
            self.registrar("=== Metricas 4 classes (%s) ===" % rede)
            self.registrar("Tempo de classificacao do teste: %.3f s" % tempo_inf)
            self.registrar("Sensibilidade media: %.4f" % metricas["sensibilidade_media"])
            self.registrar("Especificidade media: %.4f" % metricas["especificidade_media"])
            self.registrar("Acuracia: %.4f" % metricas["acuracia"])
            self._desenhar(lambda: self._plotar_matriz(matriz, ROTULOS_QUATRO, "4 classes - %s" % rede))

    def _plotar_matriz(self, matriz, rotulos, titulo):
        self.figura.clear()
        eixo = self.figura.add_subplot(111)
        imagem = eixo.imshow(matriz, cmap="Blues")
        eixo.set_xticks(range(len(rotulos)))
        eixo.set_yticks(range(len(rotulos)))
        eixo.set_xticklabels(rotulos, rotation=30, ha="right")
        eixo.set_yticklabels(rotulos)
        eixo.set_xlabel("Predito")
        eixo.set_ylabel("Verdadeiro")
        eixo.set_title("Matriz de confusao - %s" % titulo)
        limite = matriz.max() / 2.0 if matriz.max() > 0 else 0.5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                eixo.text(
                    j, i, str(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > limite else "black",
                )
        self.figura.colorbar(imagem, ax=eixo, fraction=0.046)
        self.canvas.draw()

    def acao_gradcam(self):
        rede = self.var_rede.get()
        binario = self.var_modo.get() == "binaria"
        chave = self._chave(rede, binario)
        if chave not in self.modelos:
            messagebox.showinfo("Aviso", "Treine ou carregue os pesos desta configuracao primeiro.")
            return
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("Todos", "*.*")]
        )
        if not caminho:
            return
        imagem = carregar_imagem_cinza(caminho)
        if imagem is None:
            messagebox.showerror("Erro", "Nao foi possivel ler a imagem.")
            return
        modelo, usar_segmentacao = self.modelos[chave]
        tensor = preparar_tensor(imagem, usar_segmentacao)
        grad = GradCAM(modelo, ultima_camada_convolucional(modelo))
        classe, mapa = grad.gerar(tensor)
        if binario:
            nome_classe = ROTULOS_BINARIOS[classe]
        else:
            nome_classe = "BI-RADS %s" % ROTULOS_QUATRO[classe]
        if usar_segmentacao:
            _, base = segmentar_mama(imagem)
        else:
            base = imagem
        sobreposta = sobrepor_gradcam(base, mapa)
        self.figura.clear()
        eixo1 = self.figura.add_subplot(1, 2, 1)
        eixo1.imshow(base, cmap="gray")
        eixo1.set_title("Entrada classificada como:\n%s" % nome_classe)
        eixo1.axis("off")
        eixo2 = self.figura.add_subplot(1, 2, 2)
        eixo2.imshow(sobreposta)
        eixo2.set_title("Grad-CAM (%s)" % rede)
        eixo2.axis("off")
        self.canvas.draw()
        self.registrar("Imagem classificada como %s (rede %s)." % (nome_classe, rede))


if __name__ == "__main__":
    Aplicacao().mainloop()
