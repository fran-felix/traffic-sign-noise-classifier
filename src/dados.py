"""Carregamento do GTSRB e do conjunto de desenvolvimento.

    treino_validacao(...)  -> GTSRB "train" dividido em treino e validação POR TRILHA
    PastaImagens(...)      -> imagens de uma pasta (dev ou teste), em ordem de nome
"""

import csv
import numpy as np
import torchvision

from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

N_CLASSES = 43

NOMES_CLASSES = [
    "limite 20", "limite 30", "limite 50", "limite 60", "limite 70", "limite 80",
    "fim limite 80", "limite 100", "limite 120", "proibido ultrapassar",
    "proibido ultrapassar caminhões", "cruzamento com preferência", "via preferencial",
    "dê a preferência", "pare", "trânsito proibido", "proibido caminhões",
    "sentido proibido", "perigo", "curva à esquerda", "curva à direita", "curva dupla",
    "lombada", "pista escorregadia", "estreitamento à direita", "obras", "semáforo",
    "pedestres", "crianças", "ciclistas", "gelo/neve", "animais silvestres",
    "fim das restrições", "vire à direita", "vire à esquerda", "siga em frente",
    "em frente ou à direita", "em frente ou à esquerda", "mantenha-se à direita",
    "mantenha-se à esquerda", "rotatória", "fim proibido ultrapassar",
    "fim proibido ultrapassar caminhões",
]


class ListaImagens(Dataset):
    """Dataset a partir de uma lista [(caminho, classe), ...]."""

    def __init__(self, amostras, transform=None):
        self.amostras = list(amostras)
        self.transform = transform

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, i):
        caminho, classe = self.amostras[i]
        img = Image.open(caminho).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, classe


def treino_validacao(raiz="data", frac_validacao=0.15, seed=0,
                     transform_treino=None, transform_val=None):
    """Divide o split "train" do GTSRB em treino e validação.

    Atenção: o treino do GTSRB é formado por TRILHAS de 30 fotos da mesma placa
    física, tiradas em sequência (arquivo <trilha>_<quadro>.ppm). Uma divisão
    aleatória por imagem coloca fotos quase idênticas nos dois lados e a
    acurácia de validação fica artificialmente alta. Por isso a divisão aqui é
    feita por trilha.
    """
    amostras = torchvision.datasets.GTSRB(raiz, split="train", download=True)._samples
    trilhas = defaultdict(list)
    for caminho, classe in amostras:
        trilha = (classe, Path(caminho).name.split("_")[0])
        trilhas[trilha].append((caminho, classe))
    chaves = sorted(trilhas)
    rng = np.random.default_rng(seed)
    rng.shuffle(chaves)
    n_val = int(round(frac_validacao * len(chaves)))
    val = [a for k in chaves[:n_val] for a in trilhas[k]]
    treino = [a for k in chaves[n_val:] for a in trilhas[k]]
    return ListaImagens(treino, transform_treino), ListaImagens(val, transform_val)


def treino_completo(raiz="data", transform=None):
    """Todo o split "train" do GTSRB (para o treino final)."""
    amostras = torchvision.datasets.GTSRB(raiz, split="train", download=True)._samples
    return ListaImagens(amostras, transform)


class PastaImagens(Dataset):
    """Imagens .png de uma pasta, em ordem alfabética de nome.

    Se `rotulos` (CSV com colunas arquivo,classe) for informado, devolve
    (imagem, classe); caso contrário devolve (imagem, -1).
    """

    def __init__(self, pasta, rotulos=None, transform=None):
        self.pasta = Path(pasta)
        self.arquivos = sorted(p.name for p in self.pasta.glob("*.png"))
        self.transform = transform
        self.classes = None
        self.degradacoes = None
        if rotulos is not None:
            with open(rotulos) as f:
                linhas = {r["arquivo"]: r for r in csv.DictReader(f)}
            self.classes = [int(linhas[a]["classe"]) for a in self.arquivos]
            self.degradacoes = [linhas[a].get("degradacoes", "") for a in self.arquivos]

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        img = Image.open(self.pasta / self.arquivos[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, (self.classes[i] if self.classes is not None else -1)
