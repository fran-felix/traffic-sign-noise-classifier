# Prova Optativa — Classificação de Placas de Trânsito em Imagens Degradadas

O enunciado completo está em `enunciado.pdf`.

## Estrutura

```
enunciado.pdf          enunciado da prova
environment.yml        dependências (conda)
dev/                   conjunto de desenvolvimento: 1.000 imagens degradadas
  imagens/             000001.png, 000002.png, ...
  rotulos.csv          arquivo, classe, degradações aplicadas
teste/                 conjunto de teste: 11.630 imagens degradadas, SEM rótulo
src/
  dados.py             GTSRB (treino/validação por trilha) e leitura de pastas de imagens
  modelo.py            modelo e pré-processamento (ponto de partida)
  treino_base.py       treino mínimo, sem tratamento das degradações
  predizer.py          gera NOME_SOBRENOME_RESULTS.txt; no dev, mostra a acurácia por degradação
  visualizar.py        exemplos do dev agrupados por degradação
  verificar_entrega.py verificação do formato do arquivo de predições
data/                  GTSRB (baixado automaticamente; não versionar)
```

## Ambiente

```bash
conda env create -f environment.yml   # cria o ambiente "tpi"
conda activate tpi
```

Para quem já tem o ambiente `tpi` dos trabalhos anteriores:
`conda env update -n tpi -f environment.yml`.

No Google Colab, basta enviar esta pasta (ou o `.zip` e descompactar). O Colab já
traz PyTorch, torchvision, OpenCV e matplotlib instalados.

## Passo a passo

Todos os comandos são executados na raiz desta pasta.

```bash
# 1. ver como são as degradações
python src/visualizar.py --out dev.png

# 2. treinar o modelo base (baixa o GTSRB, ~350 MB, na primeira vez)
python src/treino_base.py

# 3. avaliar no dev: acurácia geral e por degradação
python src/predizer.py --pasta dev/imagens --rotulos dev/rotulos.csv --saida dev_RESULTS.txt

# 4. com o modelo final: predições do teste + verificação do formato
python src/predizer.py --pasta teste --saida NOME_SOBRENOME_RESULTS.txt
python src/verificar_entrega.py NOME_SOBRENOME_RESULTS.txt --pasta teste
```

Resultado esperado do modelo base (10 épocas, alguns minutos em GPU):
cerca de 88% na validação (imagens limpas) e 50% no dev (imagens degradadas).

## Onde mexer

- `src/modelo.py`: arquitetura (`criar_modelo`), pré-processamento de avaliação
  (`transform_avaliacao`) e de treino, com o augmentation (`transform_treino`).
  O `predizer.py` usa essas funções, então qualquer mudança aqui vale também na
  predição.
- `src/treino_base.py`: otimizador, épocas, scheduler etc. Para o modelo final,
  `dados.treino_completo()` devolve todo o split de treino, sem separar validação.

Lembrem: o dev serve só para medir e escolher, **nunca para treinar**; as imagens
de `teste/` servem só para gerar as predições finais; e o `split="test"` do GTSRB
não pode ser usado (veja o enunciado).
