"""Mostra exemplos do conjunto de desenvolvimento, agrupados por degradação.

Uso (na raiz do material):
    python src/visualizar.py                  # abre uma janela
    python src/visualizar.py --out dev.png    # salva em arquivo
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from dados import NOMES_CLASSES


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dev", type=Path, default=Path("dev"))
    p.add_argument("--n", type=int, default=8, help="exemplos por degradação")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    por_tipo = defaultdict(list)
    for r in csv.DictReader(open(a.dev / "rotulos.csv")):
        degs = r["degradacoes"].split("+")
        if len(degs) == 1:  # só imagens com uma única degradação, para isolar o efeito
            por_tipo[degs[0]].append(r)
    tipos = sorted(por_tipo)
    fig, eixos = plt.subplots(len(tipos), a.n, figsize=(1.6 * a.n, 1.8 * len(tipos)))
    for i, t in enumerate(tipos):
        for j in range(a.n):
            ax = eixos[i, j]
            ax.axis("off")
            if j < len(por_tipo[t]):
                r = por_tipo[t][j]
                ax.imshow(Image.open(a.dev / "imagens" / r["arquivo"]))
                ax.set_title(NOMES_CLASSES[int(r["classe"])], fontsize=6)
        eixos[i, 0].text(-0.15, 0.5, t, transform=eixos[i, 0].transAxes, ha="right",
                         va="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(left=0.12)
    if a.out:
        fig.savefig(a.out, dpi=130)
        print(f"figura salva em {a.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
