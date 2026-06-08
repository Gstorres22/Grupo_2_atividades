# Pasta de dados

Esta pasta deve conter o dataset **Credit Score Classification** do Kaggle:

https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## Como baixar

### Via API pública (sem login)

```bash
cd data
curl -L -o credit_score.zip \
  https://www.kaggle.com/api/v1/datasets/download/parisrohan/credit-score-classification
unzip credit_score.zip
```

Você deve ficar com:
- `train.csv` (≈ 30 MB, 100.000 linhas)
- `test.csv`  (≈ 15 MB)

### Criar uma amostra pequena para testes rápidos

```bash
python -c "import pandas as pd; \
  pd.read_csv('data/train.csv', nrows=5000, low_memory=False) \
    .to_csv('data/train_sample.csv', index=False)"
```

E depois rodar o treinamento com:

```bash
python -m training.train --data data/train_sample.csv
```
