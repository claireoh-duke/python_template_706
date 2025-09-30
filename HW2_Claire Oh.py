# %%
# <Crypto Data Analysis>

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

# 2. Load Datasets
btc_df = pd.read_csv("BTC.csv")
eth_df = pd.read_csv("ETH.csv")
print("Bitcoin Data:")
print(btc_df.head())
print("\nEthereum Data:")
print(eth_df.head())

# %%
# 3. Data Cleaning

# Check for missing values
print("Missing values in BTC data:\n", btc_df.isnull().sum())
print("Missing values in ETH data:\n", eth_df.isnull().sum())

# Drop duplicates
btc_df.drop_duplicates(inplace=True)
eth_df.drop_duplicates(inplace=True)

print("Cleaned BTC Shape:", btc_df.shape)
print("Cleaned ETH Shape:", eth_df.shape)

# Detect simple outliers using Z-score

z_scores_btc = np.abs(stats.zscore(btc_df.select_dtypes(include=[np.number])))
z_scores_eth = np.abs(stats.zscore(eth_df.select_dtypes(include=[np.number])))
btc_df = btc_df[(z_scores_btc < 3).all(axis=1)]
eth_df = eth_df[(z_scores_eth < 3).all(axis=1)]
print("BTC Shape after outlier removal:", btc_df.shape)
print("ETH Shape after outlier removal:", eth_df.shape)

# %%
# 4. Inspect Data

# Data Overview
print("BTC Data Info:")
print(btc_df.info())
print("\nETH Data Info:")
print(eth_df.info())

# %%
# Statistical Summary
print("BTC Statistical Summary:")
print(btc_df.describe())
print("\nETH Statistical Summary:")
print(eth_df.describe())

# %%
# 5. Basic Filtering and Grouping
# Filter data for specific date ranges
print(btc_df[btc_df["date"] > "2022-12-08"].head(), "\n")
print(eth_df[eth_df["date"] > "2010-12-23"].head(), "\n")
# Filter data for high close prices
max_btc = btc_df["close"].max()
max_eth = eth_df["close"].max()

print("BTC entries with max close price:\n", btc_df[btc_df["close"] == max_btc], "\n")
print("ETH entries with max close price:\n", eth_df[eth_df["close"] == max_eth])

# %%
# Group by year and average close price
btc_df["year"] = pd.to_datetime(btc_df["date"]).dt.year
eth_df["year"] = pd.to_datetime(eth_df["date"]).dt.year
btc_yearly_avg = btc_df.groupby("year")["close"].mean().reset_index()
eth_yearly_avg = eth_df.groupby("year")["close"].mean().reset_index()
print("BTC Yearly Average Close Prices:\n", btc_yearly_avg, "\n")
print("ETH Yearly Average Close Prices:\n", eth_yearly_avg)

# %%
# 6. ML Modeling


# Feature Engineering
# 1) Lag returns
def lag_return(btc_df, eth_df):
    btc_df["ret_1d"] = np.log(btc_df["close"] / btc_df["close"].shift(1))
    eth_df["ret_1d"] = np.log(eth_df["close"] / eth_df["close"].shift(1))
    btc_df["ret_3d"] = np.log(btc_df["close"] / btc_df["close"].shift(3))
    eth_df["ret_3d"] = np.log(eth_df["close"] / eth_df["close"].shift(3))
    btc_df["ret_7d"] = np.log(btc_df["close"] / btc_df["close"].shift(7))
    eth_df["ret_7d"] = np.log(eth_df["close"] / eth_df["close"].shift(7))


lag_return(btc_df, eth_df)
# 2) Moving averages
btc_df["ma_7"] = btc_df["close"].rolling(window=7).mean()
eth_df["ma_7"] = eth_df["close"].rolling(window=7).mean()
btc_df["ma_21"] = btc_df["close"].rolling(window=21).mean()
eth_df["ma_21"] = eth_df["close"].rolling(window=21).mean()
# 3) Fluctuations
btc_df["high_low_diff"] = btc_df["high"] - btc_df["low"]
eth_df["high_low_diff"] = eth_df["high"] - eth_df["low"]
btc_df["open_close_diff"] = btc_df["open"] - btc_df["close"]
eth_df["open_close_diff"] = eth_df["open"] - eth_df["close"]


# 4) RSI (Relative Strength Index)
def rsi(btc_df, eth_df):
    delta_btc = btc_df["close"].diff()
    gain_btc = (delta_btc.where(delta_btc > 0, 0)).rolling(window=14).mean()
    loss_btc = (-delta_btc.where(delta_btc < 0, 0)).rolling(window=14).mean()
    rs_btc = gain_btc / loss_btc
    btc_df["rsi"] = 100 - (100 / (1 + rs_btc))
    delta_eth = eth_df["close"].diff()
    gain_eth = (delta_eth.where(delta_eth > 0, 0)).rolling(window=14).mean()
    loss_eth = (-delta_eth.where(delta_eth < 0, 0)).rolling(window=14).mean()
    rs_eth = gain_eth / loss_eth
    eth_df["rsi"] = 100 - (100 / (1 + rs_eth))


rsi(btc_df, eth_df)

# Remove NaN values from feature engineering
btc_df.dropna(inplace=True)
eth_df.dropna(inplace=True)

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
x_btc = btc_df[features]
y_btc = btc_df["close"].shift(-5)  # Predicting 5 days ahead
x_eth = eth_df[features]
y_eth = eth_df["close"].shift(-5)  # Predicting 5 days ahead

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


x_btc_train, x_btc_test, y_btc_train, y_btc_test = train_test_split(
    x_btc, y_btc, test_size=0.2, shuffle=False
)
x_eth_train, x_eth_test, y_eth_train, y_eth_test = train_test_split(
    x_eth, y_eth, test_size=0.2, shuffle=False
)

# Model Training (XGBoost)

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
# 8. Visualization
# 1) Time series plot
btc_df["year"] = pd.to_datetime(btc_df["date"]).dt.year
eth_df["year"] = pd.to_datetime(eth_df["date"]).dt.year
btc_yearly = btc_df.groupby("year")["close"].mean().reset_index()
btc_yearly["crypto"] = "BTC"
eth_yearly = eth_df.groupby("year")["close"].mean().reset_index()
eth_yearly["crypto"] = "ETH"
combined_yearly = pd.concat([btc_yearly, eth_yearly])
fig = px.line(
    combined_yearly,
    x="year",
    y="close",
    color="crypto",
    title="BTC vs ETH Yearly Average Closing Price",
)
fig.update_layout(xaxis_title="Year", yaxis_title="Average Close Price (USD)")

fig.show()

# 2)
plt.figure(figsize=(12, 6))
plt.plot(y_btc_test.values, label="Actual BTC", color="blue")
plt.plot(btc_preds, label="Predicted BTC", color="orange")
plt.title("BTC Actual vs Predicted Prices (5 days ahead)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_eth_test.values, label="Actual ETH", color="blue")
plt.plot(eth_preds, label="Predicted ETH", color="orange")
plt.title("ETH Actual vs Predicted Prices (5 days ahead)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# %%
