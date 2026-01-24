# 🏠 House Price Prediction - Kaggle Competition

Machine Learning project for predicting house prices using ensemble methods.

## 📊 Project Results

| Notebook | Model | Validation RMSE | R² Score | Improvement |
|----------|-------|-----------------|----------|-------------|
| 03_modeling | Ridge (Baseline) | 0.1527 | 0.8617 | - |
| 04_feature_engineering | Ridge + Features | 0.1439 | 0.8772 | +5.8% |
| 05_advanced_models | XGBoost | 0.1374 | 0.8880 | +10.0% |
| 05_advanced_models | LightGBM | 0.1342 | 0.8931 | +12.1% |
| **06_ensemble** | **Weighted Ensemble** | **0.1308** | **0.8985** | **+14.3%** |

**🏆 Best Model:** Stacked Ensemble with **RMSE 0.1308** (14.3% improvement over baseline)

---

## 🚀 Project Structure

```
Housing_Prices/
├── data/
│   ├── train.csv                              # Training data
│   ├── test.csv                               # Test data
│   ├── submission.csv                         # Baseline submission
│   ├── submission_feature_engineering.csv     # Feature Engineering submission
│   ├── submission_advanced_models.csv         # Advanced Models submission
│   └── submission_ensemble.csv                # Ensemble submission (BEST!)
│
├── notebooks/
│   ├── 01_eda.ipynb                          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb                # Data Preprocessing
│   ├── 03_modeling.ipynb                     # Baseline Models
│   ├── 04_feature_engineering.ipynb          # Feature Engineering
│   ├── 05_advanced_models.ipynb              # XGBoost & LightGBM
│   └── 06_ensemble.ipynb                     # Ensemble Methods
│
├── src/
│   ├── preprocessing.py                      # Preprocessing functions
│
├── .venv/                                    # Virtual Environment (Python 3.14.2)
├── requirements.txt                          # Python Dependencies
├── .gitignore                               # Git Ignore File
└── README.md                                # Project Documentation
```

---

## 🛠️ Tech Stack

- **Python:** 3.14.2
- **Package Manager:** uv
- **ML Libraries:**
  - scikit-learn
  - XGBoost
  - LightGBM
- **Data Processing:**
  - pandas
  - numpy
- **Visualization:**
  - matplotlib
  - seaborn

---

## 📈 ML Pipeline

### 1. **Exploratory Data Analysis (EDA)**
- **Data:** 1,460 training houses, 1,459 test houses
- **Target:** SalePrice (right-skewed: Skewness 1.88, Kurtosis 6.54)
- **Features:** 80 features (43 categorical, 37 numerical)
- **Missing Values:** 19 features with NAs (PoolQC: 99.5%, MiscFeature: 96.3%, Alley: 93.8%)
- **Outliers identified:** 2 houses (IDs 523, 1298)

### 2. **Preprocessing**
```python
# Main steps:
✓ Outlier Removal (IDs: 523, 1298)
✓ Missing Value Imputation
  - Categorical: "None" for absent features
  - Numerical: 0 or Median/Neighborhood-specific median
✓ Log-Transformation of target variable
✓ Ordinal feature encoding
✓ One-Hot Encoding (211 features)
```

**Result:** Training Shape (1,458, 79) → (1,458, 211)

### 3. **Feature Engineering** (+17 Features)

#### New Features Created:
1. **TotalSF** = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
2. **TotalBath** = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath
3. **TotalPorchSF** = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
4. **HouseAge** = YrSold - YearBuilt
5. **YearsSinceRemod** = YrSold - YearRemodAdd
6. **IsRemodeled** = (YearBuilt != YearRemodAdd)
7. **OverallQual_TotalSF** = OverallQual × TotalSF
8. **OverallQual_GrLivArea** = OverallQual × GrLivArea
9. **HasPool** = (PoolArea > 0)
10. **Has2ndFloor** = (2ndFlrSF > 0)
11. **HasGarage** = (GarageArea > 0)
12. **HasBsmt** = (TotalBsmtSF > 0)
13. **HasFireplace** = (Fireplaces > 0)
14. **TotalQuality** = OverallQual + OverallCond
15. **LotArea_per_GrLivArea** = LotArea / (GrLivArea + 1)
16. **GarageScore** = GarageQual + GarageCond
17. **KitchenPerRoom** = KitchenAbvGr / (TotRmsAbvGrd + 1)

**Result:** 228 features total (211 + 17 new)

**Top Features by Importance:**
- Neighborhood_StoneBr (0.092)
- Neighborhood_CrawFor (0.083)
- Neighborhood_NridgHt (0.080)
- OverallQual (0.057)
- GarageCars (0.047)
- TotalQuality (0.046)

### 4. **Baseline Models**

```python
# Ridge Regression (Best Baseline)
Ridge(alpha=10.0)
→ RMSE: 0.1527, R²: 0.8617

# Random Forest
RandomForestRegressor(n_estimators=100)
→ RMSE: 0.1575, R²: 0.8528

# Linear Regression
→ RMSE: 0.2373, R²: 0.6659
```

### 5. **Advanced Models**

#### XGBoost Configuration:
```python
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.001,
    reg_alpha=0.01,
    reg_lambda=1.0
)
→ RMSE: 0.1374
```

#### LightGBM Configuration:
```python
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=20,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8
)
→ RMSE: 0.1342
```

### 6. **Ensemble Methods** 🎯

#### Models Used:
1. Ridge (Baseline)
2. XGBoost v1
3. LightGBM
4. XGBoost v2 (deeper: max_depth=5, learning_rate=0.03)

#### Ensemble Strategies:

**Simple Average:**
```python
predictions = (pred_ridge + pred_xgb + pred_lgb + pred_xgb2) / 4
→ RMSE: 0.1319, R²: 0.8968
```

**Weighted Average:**
```python
# Weights based on inverse RMSE
weights = [0.239, 0.250, 0.256, 0.256]
predictions = weighted_sum(all_predictions, weights)
→ RMSE: 0.1319, R²: 0.8968
```

**Stacking with Ridge Meta-Learner (BEST!):**
```python
meta_learner = Ridge(alpha=1.0)
→ RMSE: 0.1308, R²: 0.8985
```

---

## 🔧 Installation & Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd Housing_Prices
```

### 2. Create Virtual Environment
```bash
# Using uv (recommended)
uv venv .venv --python 3.14.2
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Alternative: Using pip
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Notebooks in Order:
```bash
# 1. Exploratory Data Analysis
jupyter notebook notebooks/01_eda.ipynb

# 2. Data Preprocessing
jupyter notebook notebooks/02_preprocessing.ipynb

# 3. Baseline Models
jupyter notebook notebooks/03_modeling.ipynb

# 4. Feature Engineering
jupyter notebook notebooks/04_feature_engineering.ipynb

# 5. Advanced Models
jupyter notebook notebooks/05_advanced_models.ipynb

# 6. Ensemble Methods (Final)
jupyter notebook notebooks/06_ensemble.ipynb
```

### Use Preprocessing Module:
```python
from src.preprocessing import preprocess_data

# Load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Preprocess
train_processed, test_processed, y_log, train_ids, test_ids = preprocess_data(
    train, test, remove_outliers_flag=True
)
```

---

## 📊 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Log transform SalePrice** | Target is right-skewed (1.88) → normalizes distribution |
| **Remove outliers (IDs 523, 1298)** | Large GrLivArea but low price → hurts model performance |
| **Neighborhood-specific median for LotFrontage** | Better than global median due to neighborhood variance |
| **One-hot encoding categorical features** | 43 categorical features → 211 columns after encoding |
| **80/20 train/validation split** | Stratified by price range with random_state=42 |
| **Ensemble of 4 models** | Reduces variance and improves generalization |

---

## 🎯 Model Performance Progression

```
Baseline Ridge (0.1527)
         ↓ (+5.8%)
Ridge + Features (0.1439)
         ↓ (+4.1%)
XGBoost (0.1374)
         ↓ (+2.3%)
LightGBM (0.1342)
         ↓ (+1.7%)
Stacked Ensemble (0.1308) ← BEST!
```

**Total Improvement:** 14.3% from baseline to ensemble

---

## 📁 Submissions Created

1. **submission.csv** - Ridge baseline (RMSE 0.1527)
2. **submission_feature_engineering.csv** - Ridge + features (RMSE 0.1439)
3. **submission_advanced_models.csv** - XGBoost/LightGBM (RMSE ~0.134)
4. **submission_ensemble.csv** - Stacked ensemble (RMSE 0.1308) ⭐

---

## 🚀 Next Steps & Improvements

### Potential Enhancements:
- [ ] Hyperparameter tuning
- [ ] Cross-validation for more robust evaluation
- [ ] Feature selection to reduce dimensionality
- [ ] Try additional models (CatBoost, Neural Networks)

---

## 📝 Dependencies

```txt
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
jupyter
```

Install all at once:
```bash
uv pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

---

## 🤝 Contributions

Raphael Benzenhöfer, Klaas Coerdes, Suneet Khaneja

---

## 📄 License

This project is open source and available under the MIT License.

---

## Acknowledgments

- **Kaggle Competition:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Dataset:** Ames Housing Dataset
- **Tools:** Python, scikit-learn, XGBoost, LightGBM
