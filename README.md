# House Price Prediction Project

Predicting house sale prices using machine learning.

## Setup

1. Create virtual environment:
```bash
uv venv --python 3.14
```

2. Activate:
```bash
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Project Structure
```
Housing_Prices/
├── .venv/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
├── notebooks/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Setup

Download the dataset from Kaggle:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Place the files in the `data/` folder:
- train.csv
- test.csv
- submission.csv