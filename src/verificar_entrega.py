"""Verifica o formato do arquivo de predições e da pasta de entrega.

Uso (na raiz do material):
    python src/verificar_entrega.py NOME_SOBRENOME_RESULTS.txt --pasta teste
    python src/verificar_entrega.py NOME_SOBRENOME_RESULTS.txt --pasta teste --codigo .

Confere: nome do arquivo, uma predição por linha, quantidade de linhas igual ao
número de imagens da pasta, valores inteiros de 0 a 42. Com --codigo, confere
também os arquivos obrigatórios da entrega de código.
Termina com código 0 se tudo estiver certo.
"""

import argparse
import re
import sys
from pathlib import Path

N_CLASSES = 43
OBRIGATORIOS = ["README.md", "predizer.py", "pesos"]


def check(ok, msg, erros):
    print(("  [ok]   " if ok else "  [ERRO] ") + msg)
    if not ok:
        erros.append(msg)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultados", type=Path)
    p.add_argument("--pasta", type=Path, required=True, help="pasta com as imagens avaliadas")
    p.add_argument("--codigo", type=Path, default=None, help="pasta com o código entregue")
    a = p.parse_args()
    erros = []

    print("1. Arquivo de predições")
    check(re.fullmatch(r"[A-Z]+(_[A-Z]+)+_RESULTS\.txt", a.resultados.name) is not None,
          f"nome '{a.resultados.name}' no formato NOME_SOBRENOME_RESULTS.txt (maiúsculas, sem acento)",
          erros)
    n_imagens = len(list(a.pasta.glob("*.png")))
    check(n_imagens > 0, f"{n_imagens} imagens .png em {a.pasta}", erros)
    if check(a.resultados.is_file(), f"{a.resultados} existe", erros):
        linhas = a.resultados.read_text().splitlines()
        while linhas and linhas[-1] == "":
            linhas.pop()
        check(len(linhas) == n_imagens, f"{len(linhas)} linhas (esperado {n_imagens})", erros)
        ruins = [i + 1 for i, v in enumerate(linhas)
                 if not (v.strip().isdigit() and 0 <= int(v) < N_CLASSES and v == v.strip())]
        check(not ruins, "todas as linhas com um único inteiro entre 0 e 42"
              + (f" (problemas nas linhas {ruins[:10]}...)" if ruins else ""), erros)

    if a.codigo:
        print("2. Código")
        for nome in OBRIGATORIOS:
            achados = [f for f in a.codigo.rglob(nome) if "data" not in f.parts]
            check(bool(achados), nome + (f" ({achados[0].relative_to(a.codigo)})" if achados else ""),
                  erros)

    print("\nOK: entrega no formato correto." if not erros else f"\n{len(erros)} problema(s).")
    sys.exit(1 if erros else 0)


if __name__ == "__main__":
    main()
