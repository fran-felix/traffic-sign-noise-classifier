"""Gera o arquivo de predições para uma pasta de imagens (dev ou teste).

Uso (na raiz do material):
    # conjunto de desenvolvimento, com acurácia geral e por degradação
    python src/predizer.py --pasta dev/imagens --rotulos dev/rotulos.csv --saida dev_RESULTS.txt

    # conjunto de teste (entrega)
    python src/predizer.py --pasta teste --saida NOME_SOBRENOME_RESULTS.txt

O arquivo de saída tem uma classe (0 a 42) por linha, na ordem alfabética dos
nomes das imagens (000001.png, 000002.png, ...).
"""

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dados import PastaImagens
from modelo import criar_modelo, transform_avaliacao


@torch.no_grad()
def predizer(modelo, dataset, device, batch=256):
  modelo.eval()
  preds = []
  for x, _ in DataLoader(dataset, batch_size=batch):
    preds += modelo(x.to(device)).argmax(1).tolist()
  return preds


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--pasta", required=True)
  p.add_argument("--saida", required=True)
  p.add_argument("--pesos", default="pesos/modelo.pt")
  p.add_argument("--rotulos", default=None, help="CSV de rótulos (só para o dev)")
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  a = p.parse_args()

  modelo = criar_modelo()
  modelo.load_state_dict(torch.load(a.pesos, map_location="cpu", weights_only=True))
  modelo.to(a.device)
  dataset = PastaImagens(a.pasta, a.rotulos, transform=transform_avaliacao())
  preds = predizer(modelo, dataset, a.device)

  with open(a.saida, "w") as f:
      f.write("\n".join(map(str, preds)) + "\n")
  print(f"{len(preds)} predições salvas em {a.saida}")

  if dataset.classes is not None:
    certo = [p == y for p, y in zip(preds, dataset.classes)]
    print(f"acurácia: {sum(certo) / len(certo):.4f}")
    por_tipo = defaultdict(list)
    for c, degs in zip(certo, dataset.degradacoes):
      for d in degs.split("+"):
        por_tipo[d].append(c)
    print("acurácia por degradação (imagens que a contêm):")
    for d in sorted(por_tipo, key=lambda d: sum(por_tipo[d]) / len(por_tipo[d])):
      v = por_tipo[d]
      print(f"  {d:16s} {sum(v) / len(v):.3f}  ({len(v)} imagens)")


if __name__ == "__main__":
  main()
