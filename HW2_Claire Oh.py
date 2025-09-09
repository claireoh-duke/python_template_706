# %%
# <Crypto Data Analysis>

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Load Datasets
df1 = pd.read_csv("/Users/otting/Desktop/Data/BTC.csv")
df2 = pd.read_csv("/Users/otting/Desktop/Data/ETH.csv")
print("Bitcoin Data:")
print(df1.head())
print("\nEthereum Data:")
print(df2.head())

# %%
# 3. Data Cleaning

# Check for missing values
print("Missing values in BTC data:\n", df1.isnull().sum())
print("Missing values in ETH data:\n", df2.isnull().sum())

# Drop duplicates
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)

print("Cleaned BTC Shape:", df1.shape)
print("Cleaned ETH Shape:", df2.shape)

# Detect simple outliers using Z-score
from scipy import stats

z_scores_btc = np.abs(stats.zscore(df1.select_dtypes(include=[np.number])))
z_scores_eth = np.abs(stats.zscore(df2.select_dtypes(include=[np.number])))
df1 = df1[(z_scores_btc < 3).all(axis=1)]
df2 = df2[(z_scores_eth < 3).all(axis=1)]
print("BTC Shape after outlier removal:", df1.shape)
print("ETH Shape after outlier removal:", df2.shape)

# %%
# 4. Inspect Data

# Data Overview
print("BTC Data Info:")
print(df1.info())
print("\nETH Data Info:")
print(df2.info())

# %%
# Statistical Summary
print("BTC Statistical Summary:")
print(df1.describe())
print("\nETH Statistical Summary:")
print(df2.describe())

# %%
# 5. Basic Filtering and Grouping
# Filter data for specific date ranges
print(df1[df1["date"] > "2022-12-08"].head(), "\n")
print(df2[df2["date"] > "2010-12-23"].head(), "\n")
# Filter data for high close prices
max_btc = df1["close"].max()
max_eth = df2["close"].max()

print("BTC entries with max close price:\n", df1[df1["close"] == max_btc], "\n")
print("ETH entries with max close price:\n", df2[df2["close"] == max_eth])

# %%
# Group by year and average close price
df1["year"] = pd.to_datetime(df1["date"]).dt.year
df2["year"] = pd.to_datetime(df2["date"]).dt.year
btc_yearly_avg = df1.groupby("year")["close"].mean().reset_index()
eth_yearly_avg = df2.groupby("year")["close"].mean().reset_index()
print("BTC Yearly Average Close Prices:\n", btc_yearly_avg, "\n")
print("ETH Yearly Average Close Prices:\n", eth_yearly_avg)

# %%
# 6. ML Modeling

# Feature Engineering
# 1) Lag returns
df1["ret_1d"] = np.log(df1["close"] / df1["close"].shift(1))
df2["ret_1d"] = np.log(df2["close"] / df2["close"].shift(1))
df1["ret_3d"] = np.log(df1["close"] / df1["close"].shift(3))
df2["ret_3d"] = np.log(df2["close"] / df2["close"].shift(3))
df1["ret_7d"] = np.log(df1["close"] / df1["close"].shift(7))
df2["ret_7d"] = np.log(df2["close"] / df2["close"].shift(7))
# 2) Moving averages
df1["ma_7"] = df1["close"].rolling(window=7).mean()
df2["ma_7"] = df2["close"].rolling(window=7).mean()
df1["ma_21"] = df1["close"].rolling(window=21).mean()
df2["ma_21"] = df2["close"].rolling(window=21).mean()
# 3) Fluctuations
df1["high_low_diff"] = df1["high"] - df1["low"]
df2["high_low_diff"] = df2["high"] - df2["low"]
df1["open_close_diff"] = df1["open"] - df1["close"]
df2["open_close_diff"] = df2["open"] - df2["close"]
# 4) RSI (Relative Strength Index)
delta_btc = df1["close"].diff()
gain_btc = (delta_btc.where(delta_btc > 0, 0)).rolling(window=14).mean()
loss_btc = (-delta_btc.where(delta_btc < 0, 0)).rolling(window=14).mean()
rs_btc = gain_btc / loss_btc
df1["rsi"] = 100 - (100 / (1 + rs_btc))
delta_eth = df2["close"].diff()
gain_eth = (delta_eth.where(delta_eth > 0, 0)).rolling(window=14).mean()
loss_eth = (-delta_eth.where(delta_eth < 0, 0)).rolling(window=14).mean()
rs_eth = gain_eth / loss_eth
df2["rsi"] = 100 - (100 / (1 + rs_eth))

# Remove NaN values from feature engineering
df1.dropna(inplace=True)
df2.dropna(inplace=True)

# %%
# features and target variable
features = [
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "ma_7",
    "ma_21",
    "high_low_diff",
    "open_close_diff",
    "rsi",
]
x_btc = df1[features]
y_btc = df1["close"].shift(-5)  # Predicting 5 days ahead
x_eth = df2[features]
y_eth = df2["close"].shift(-5)  # Predicting 5 days ahead

# Romove NaN values from x, y
x_btc = x_btc.iloc[:-5]
y_btc = y_btc.dropna()
x_eth = x_eth.iloc[:-5]
y_eth = y_eth.dropna()

# Index reset
x_btc.reset_index(drop=True, inplace=True)
y_btc.reset_index(drop=True, inplace=True)

# %%
# train-test split
from sklearn.model_selection import train_test_split

x_btc_train, x_btc_test, y_btc_train, y_btc_test = train_test_split(
    x_btc, y_btc, test_size=0.2, shuffle=False
)
x_eth_train, x_eth_test, y_eth_train, y_eth_test = train_test_split(
    x_eth, y_eth, test_size=0.2, shuffle=False
)

# Model Training (XGBoost)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# BTC Model
btc_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42
)
btc_model.fit(x_btc_train, y_btc_train)
btc_preds = btc_model.predict(x_btc_test)
btc_mse = mean_squared_error(y_btc_test, btc_preds)
btc_rsme = np.sqrt(btc_mse)
btc_r2 = r2_score(y_btc_test, btc_preds)
print(f"BTC Model - MSE: {btc_mse}, RSME: {btc_rsme}, R2: {btc_r2}")

# ETH Model
eth_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42
)
eth_model.fit(x_eth_train, y_eth_train)
eth_preds = eth_model.predict(x_eth_test)
eth_mse = mean_squared_error(y_eth_test, eth_preds)
eth_rsme = np.sqrt(eth_mse)
eth_r2 = r2_score(y_eth_test, eth_preds)
print(f"ETH Model - MSE: {eth_mse}, RSME: {eth_rsme}, R2: {eth_r2}")

# %%
# 7. Feature Importance Visualization
importances_btc = btc_model.feature_importances_
importances_eth = eth_model.feature_importances_
features = x_btc.columns

x = np.arange(len(features))
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(
    x - width / 2, importances_btc, width=width, color="blue", alpha=0.7, label="BTC"
)
plt.bar(
    x + width / 2, importances_eth, width=width, color="orange", alpha=0.7, label="ETH"
)

plt.xticks(x, features, rotation=45)
plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.title("Feature Importance for BTC and ETH Models")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 8. Interactive Visualization with Plotly
import plotly.express as px

feature = "ma_7"

fig = px.scatter(
    df1,
    x=feature,
    y="close",
    title=f"BTC: {feature} vs Close Price",
    labels={feature: feature, "close": "Close Price"},
    trendline="ols",
)
fig.show()

# %%
