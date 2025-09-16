import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# 1. Load Data for all tests
@pytest.fixture
def data_load():
    df1 = pd.read_csv("BTC.csv")
    df2 = pd.read_csv("ETH.csv")
    return df1, df2

# 2. Testing
# [Test 1] Data Loading
def test_data_loading(data_load):
    df1, df2 = data_load
    
    # 1) Not Empty
    assert not df1.empty, "BTC.csv is empty"
    assert not df2.empty, "ETH.csv is empty"
    
    # 2) Columns exist
    name = "close"
    assert name in df1.columns, f"Column '{name}' not found in BTC data"
    assert name in df2.columns, f"Column '{name}' not found in ETH data"
    
    # 3) No duplicate dates
    assert df1['date'].duplicated().sum() == 0, "BTC has duplicated date"
    assert df2['date'].duplicated().sum() == 0, "ETH has duplicated date"

# [Test 2] Data Filtering
def test_data_filtering(data_load):
    df1, df2 = data_load
    
    # 1) Drop Duplicates
    df1_clean = df1.drop_duplicates()
    df2_clean = df2.drop_duplicates()
    assert df1_clean.shape[0] <= df1.shape[0]
    assert df2_clean.shape[0] <= df2.shape[0]
    
    # 2) Check nulls
    assert df1_clean['close'].isnull().sum() == 0, "BTC 'close' has null values"
    assert df2_clean['close'].isnull().sum() == 0, "ETH 'close' has null values"
    
    # 3) Check Realistic Price range
    assert (df1['high'] >= 0).all(), "BTC 'high' contains negative values"
    assert (df2['high'] >= 0).all(), "ETH 'high' contains negative values"

# [Test 3] ML
@pytest.fixture
def feature_data(data_load):
    df1, df2 = data_load
    
    # Feature engineering
    df1["ret_1d"] = np.log(df1["close"] / df1["close"].shift(1))
    df1["ma_7"] = df1["close"].rolling(7).mean()
    df1.dropna(inplace=True)
    
    df2["ret_1d"] = np.log(df2["close"] / df2["close"].shift(1))
    df2["ma_7"] = df2["close"].rolling(7).mean()
    df2.dropna(inplace=True)
    
    return df1, df2

def test_btc_model_performance(feature_data):
    df1, df2 = feature_data  
    
    features = ["ret_1d", "ma_7"]
    y_btc = df1["close"].shift(-5).dropna()
    x_btc = df1[features].iloc[:len(y_btc)]
    
    model = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(x_btc, y_btc)
    preds = model.predict(x_btc)
    
    r2 = r2_score(y_btc, preds)
    mse = mean_squared_error(y_btc, preds)
    
    assert 0 <= r2 <= 1, f"R2 out of bounds: {r2}"
    assert mse < 1e8, f"MSE too large: {mse}"

def test_eth_model_performance(feature_data):
    df1, df2 = feature_data  
    
    features = ["ret_1d", "ma_7"]
    y_eth = df2["close"].shift(-5).dropna()
    x_eth = df2[features].iloc[:len(y_eth)]
    
    model = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(x_eth, y_eth)
    preds = model.predict(x_eth)
    
    r2 = r2_score(y_eth, preds)
    mse = mean_squared_error(y_eth, preds)
    
    assert 0 <= r2 <= 1, f"R2 out of bounds: {r2}"
    assert mse < 1e8, f"MSE too large: {mse}"
