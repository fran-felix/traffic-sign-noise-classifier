"""Visualiza o efeito de cada degradação sobre uma imagem.

Uso na raiz do projeto:
  python src/visualizar_ruidos.py
  python src/visualizar_ruidos.py caminho/para/imagem.png
  python src/visualizar_ruidos.py --out degradacoes.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

from noise import NOISE


def criar_painel(imagem: Image.Image) -> Figure:
  imagens = [("original", imagem.convert("RGB"))]

  for nome, degradacao in NOISE.items():
    if degradacao is not None:
      imagens.append((nome, degradacao(imagem.copy())))

  colunas = 5
  linhas = math.ceil(len(imagens) / colunas)
  figura, eixos = plt.subplots(
    linhas,
    colunas,
    figsize=(2.4 * colunas, 2.8 * linhas),
    squeeze=False,
  )

  for eixo in eixos.flat:
    eixo.axis("off")

  for eixo, (nome, resultado) in zip(eixos.flat, imagens):
    eixo.imshow(resultado)
    eixo.set_title(nome)

  figura.tight_layout()
  return figura


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__ or "Visualiza o efeito de cada degradação sobre uma imagem."
  )
  parser.add_argument(
    "imagem",
    nargs="?",
    type=Path,
    default=Path("data/gtsrb/GTSRB/Training/00001/00000_00000.ppm"),
    help="imagem original de entrada (padrão: primeira imagem da classe 00000 em data/)",
  )
  parser.add_argument("--out", type=Path, help="salva o painel nesse arquivo")
  args = parser.parse_args()

  with Image.open(args.imagem) as imagem:
    figura = criar_painel(imagem)

  if args.out:
    figura.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"painel salvo em {args.out}")
  else:
    plt.show()


if __name__ == "__main__":
  main()
