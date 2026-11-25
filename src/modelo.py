"""Modelo e pré-processamento usados pelo treino_base.py e pelo predizer.py.

Este é apenas um ponto de partida: uma CNN pequena, com entrada 32x32 e
nenhum tratamento das degradações. Modifiquem à vontade, mas lembrem que o
predizer.py usa `criar_modelo()` e `transform_avaliacao()` deste arquivo; se
mudarem a arquitetura ou o pré-processamento, mantenham esses dois nomes (ou
adaptem o predizer.py).
"""

import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms

from dados import N_CLASSES
from noise import NOISE, rng

TAMANHO = 32
MEDIA = (0.5, 0.5, 0.5)
DESVIO = (0.5, 0.5, 0.5)

# Informações do backbone (ResNet18)
pesos_imagenet = torchvision.models.ResNet18_Weights.DEFAULT
MEDIA_IMGNET = (0.485, 0.456, 0.406)   # normalização da ImageNet
DESVIO_IMGNET = (0.229, 0.224, 0.225)

def transform_avaliacao():
  """Pré-processamento aplicado na validação, no dev e no teste."""
  return transforms.Compose([
    transforms.Resize((TAMANHO, TAMANHO)),
    transforms.ToTensor(),
    #transforms.Normalize(MEDIA, DESVIO),
    transforms.Normalize(MEDIA_IMGNET, DESVIO_IMGNET),
  ])

def degradar(image):
  degradacoes = list(NOISE.values())
  sorteadas = rng.choice(len(degradacoes), size=3, replace=False)

  for idx in sorteadas:
    image = degradacoes[int(idx)](image)

  return image

def transform_treino():
  """Pré-processamento do treino. Aqui é o lugar do data augmentation."""
  return transforms.Compose([degradar, transform_avaliacao()])

class TrafficHead(nn.Module):
  def __init__(self):
    super().__init__()
    #self.avgpool = nn.AvgPool2d(2, stride=2) # 8x8x512 -> 4x4x512

    self.mlp = nn.Sequential(nn.Flatten(),
                            nn.Linear(512, 256), nn.ReLU(),
                            nn.Linear(256, N_CLASSES))
    
    '''self.mlp = nn.Sequential(nn.Flatten(),
                            nn.Linear(512 * 4 * 4, 256), nn.ReLU(),
                            nn.Linear(256, N_CLASSES))'''

    '''self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Linear(512 * 8 * 8, 1024), nn.ReLU(),
                                nn.Linear(1024, 512), nn.ReLU(),
                                nn.Linear(512, N_CLASSES))'''

  def forward(self, x):
    #x = self.avgpool(x)
    x = self.mlp(x)
    return x
    

class TrafficNoiseNet(nn.Module):
  def __init__(self):
    super().__init__()
    resnet = torchvision.models.resnet18(weights=pesos_imagenet)
    self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    self.head = TrafficHead()

    # Congela pesos do backbone
    for p in self.backbone.parameters(): p.requires_grad = False

  def forward(self, x):
    x = self.head(self.backbone(x))
    return x
    

def bloco(entrada, saida):
  return nn.Sequential(
    nn.Conv2d(entrada, saida, 3, padding=1), nn.ReLU(),
    nn.Conv2d(saida, saida, 3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
  )


def criar_modelo():
  return TrafficNoiseNet()
  
  '''return nn.Sequential(
    bloco(3, 32),    # 32 -> 16
    bloco(32, 64),   # 16 -> 8
    bloco(64, 128),  # 8 -> 4
    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
    nn.Linear(256, N_CLASSES),
  )'''


def main() -> None:
  model = TrafficNoiseNet()
  print(model.backbone.parameters)

if __name__ == "__main__":
  main()