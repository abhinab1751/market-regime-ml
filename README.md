# 📊 Market Regime Detection using Machine Learning

An end-to-end Machine Learning project that detects stock market regimes 
(Bull / Bear / Sideways) using technical indicators and XGBoost, 
deployed using Streamlit.

---

## 🧠 Problem Statement

Financial markets move in different regimes:
- 📈 Bull Market
- 📉 Bear Market
- 🔁 Sideways Market

Detecting the current regime helps:
- Quant traders
- Portfolio managers
- Algorithmic strategies
- Risk management systems

This project builds an ML pipeline to classify market regimes using historical data.

---

# 🏗️ System Architecture
            ┌─────────────────────┐
            │   User Input (UI)   │
            │  (Ticker Selection) │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Data Fetching      │
            │  (yfinance API)     │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ Feature Engineering │
            │ - Returns           │
            │ - Volatility        │
            │ - Moving Averages   │
            │ - Momentum          │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ Trained ML Model    │
            │ (XGBoost Classifier)│
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │  Regime Prediction  │
            │  Bull / Bear / Flat │
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ Visualization Layer │
            │  - Charts           │
            │  - SHAP Analysis    │
            └─────────────────────┘

# 📂 Project Structure
market_regime_ml/
│
├── data/
│   └── raw_data.csv
│
├── features/
│   └── feature_engineering.py
│
├── models/
│   ├── train.py
│   ├── evaluate.py
│   └── saved_model.pkl
│
├── labeling/
│   └── label_generator.py
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   └── experimentation.ipynb
│
└── README.md


---

# ⚙️ Machine Learning Pipeline

### 1️⃣ Data Collection
- Historical data downloaded using `yfinance`

### 2️⃣ Feature Engineering
- Log Returns
- Rolling Volatility
- Moving Averages
- Momentum Indicators

### 3️⃣ Label Generation
Market regime defined based on return + volatility thresholds.

### 4️⃣ Model Training
- Model: XGBoost Classifier
- Multi-class classification
- Train/Test split
- Feature importance extraction

### 5️⃣ Model Interpretation
- SHAP value analysis
- Feature importance plots

---

# 📊 Model Used

- XGBoost Classifier
- Handles non-linearity
- Works well for tabular financial data
- Robust against noise

---

# 💻 Deployment

This project is deployed using:

- Streamlit (Frontend + Hosting)
- GitHub (Version Control)

---

# 🛠️ Installation (Local Setup)

```bash
git clone https://github.com/abhinab1751/market-regime-ml.git
cd market-regime-ml
pip install -r requirements.txt
streamlit run app/streamlit_app.py
