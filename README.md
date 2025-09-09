# 📊 Crypto Data Analysis
This project explores Bitcoin (BTC) and Ethereum (ETH) price data through data cleaning, feature engineering, exploratory data analysis (EDA), and machine learning modeling. The goal is to understand price movements, engineer predictive features, and train models to forecast short-term (5-day ahead) closing prices.

# 📂 Project Structure
crypto-data-analysis/
│
├── BTC.csv                  # Bitcoin historical price data
├── ETH.csv                  # Ethereum historical price data
├── crypto_analysis.ipynb    # Main Jupyter Notebook (or .py script)
└── README.md                # Project documentation

# 🧰 Tech Stack
Programming: Python 3.12.11
Libraries:
pandas, numpy → Data manipulation & analysis
matplotlib, plotly.express → Visualization
scipy → Outlier detection (Z-score)
xgboost → Machine Learning (regression model)
scikit-learn → Train/test split, metrics, evaluation
