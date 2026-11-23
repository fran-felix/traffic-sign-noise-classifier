"""Treino de referência MÍNIMO: CNN pequena, sem nenhum tratamento das degradações.

Serve para mostrar o pipeline completo (dados -> treino -> avaliação -> pesos)
e para evidenciar o problema: a acurácia na validação (imagens limpas) é alta,
mas cai muito no conjunto de desenvolvimento (imagens degradadas).

Uso (na raiz do material, com o ambiente ativo):
    python src/treino_base.py
    python src/treino_base.py --epocas 15 --device cuda
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dados import PastaImagens, treino_validacao
from modelo import criar_modelo, transform_avaliacao, transform_treino


@torch.no_grad()
def avaliar(modelo, loader, device):
    modelo.eval()
    acertos = total = 0
    perda = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        saida = modelo(x)
        perda += nn.functional.cross_entropy(saida, y, reduction="sum").item()
        acertos += (saida.argmax(1) == y).sum().item()
        total += len(y)
    return perda / total, acertos / total


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="data")
    p.add_argument("--dev", default="dev")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--saida", default="pesos/modelo.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.manual_seed(0)

    treino, val = treino_validacao(a.data, transform_treino=transform_treino(),
                                   transform_val=transform_avaliacao())
    dev = PastaImagens(Path(a.dev) / "imagens", Path(a.dev) / "rotulos.csv",
                       transform=transform_avaliacao())
    print(f"treino: {len(treino)}  validação: {len(val)}  dev: {len(dev)}")
    kw = dict(batch_size=a.batch, num_workers=a.workers)
    loader_treino = DataLoader(treino, shuffle=True, drop_last=True, **kw)
    loader_val = DataLoader(val, **kw)
    loader_dev = DataLoader(dev, **kw)

    modelo = criar_modelo().to(a.device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=a.lr)

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        t0 = time.time()
        perda_soma = acertos = total = 0
        for x, y in loader_treino:
            x, y = x.to(a.device), y.to(a.device)
            saida = modelo(x)
            perda = nn.functional.cross_entropy(saida, y)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perda_soma += perda.item() * len(y)
            acertos += (saida.argmax(1) == y).sum().item()
            total += len(y)
        perda_val, acc_val = avaliar(modelo, loader_val, a.device)
        perda_dev, acc_dev = avaliar(modelo, loader_dev, a.device)
        print(f"época {epoca:2d} | treino: perda {perda_soma / total:.3f} acc {acertos / total:.3f}"
              f" | val: perda {perda_val:.3f} acc {acc_val:.3f}"
              f" | dev: perda {perda_dev:.3f} acc {acc_dev:.3f} | {time.time() - t0:.0f}s")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), a.saida)
    print(f"pesos salvos em {a.saida}")


if __name__ == "__main__":
    main()
