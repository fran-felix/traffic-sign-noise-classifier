"""Modelo e pré-processamento usados pelo treino_base.py e pelo predizer.py.

Este é apenas um ponto de partida: uma CNN pequena, com entrada 32x32 e
nenhum tratamento das degradações. Modifiquem à vontade, mas lembrem que o
predizer.py usa `criar_modelo()` e `transform_avaliacao()` deste arquivo; se
mudarem a arquitetura ou o pré-processamento, mantenham esses dois nomes (ou
adaptem o predizer.py).
"""

import torch.nn as nn
from torchvision import transforms

from dados import N_CLASSES

TAMANHO = 32
MEDIA = (0.5, 0.5, 0.5)
DESVIO = (0.5, 0.5, 0.5)


def transform_avaliacao():
    """Pré-processamento aplicado na validação, no dev e no teste."""
    return transforms.Compose([
        transforms.Resize((TAMANHO, TAMANHO)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])


def transform_treino():
    """Pré-processamento do treino. Aqui é o lugar do data augmentation."""
    return transform_avaliacao()


def bloco(entrada, saida):
    return nn.Sequential(
        nn.Conv2d(entrada, saida, 3, padding=1), nn.ReLU(),
        nn.Conv2d(saida, saida, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
    )


def criar_modelo():
    return nn.Sequential(
        bloco(3, 32),    # 32 -> 16
        bloco(32, 64),   # 16 -> 8
        bloco(64, 128),  # 8 -> 4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        nn.Linear(256, N_CLASSES),
    )
