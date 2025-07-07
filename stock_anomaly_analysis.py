import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_stock_data(ticker="AAPL", period="7d", interval="5m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df[['Close']].dropna()
    return df

def detect_anomalies(df, window=10, threshold=2):
    df = df.copy()
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    z_score = (df['Close'] - rolling_mean) / rolling_std
    df['z_score'] = z_score
    df['anomaly'] = z_score.abs() > threshold
    df = df.dropna()
    return df

def plot_anomalies(df, ticker):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.scatter(df[df['anomaly']].index, df[df['anomaly']]['Close'],
                color='red', label='Anomaly', marker='x')
    plt.title(f"{ticker} Stock Price with Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Analyze AAPL
ticker_aapl = "AAPL"
df_aapl = fetch_stock_data(ticker_aapl)
df_aapl = detect_anomalies(df_aapl)
print("🔍 Detected anomalies in Apple (AAPL):\n", df_aapl[df_aapl['anomaly']])
plot_anomalies(df_aapl, ticker_aapl)
df_aapl[df_aapl['anomaly']].to_csv("AAPL_anomalies.csv")

# Analyze GOOGL
ticker_googl = "GOOGL"
df_googl = fetch_stock_data(ticker_googl)
df_googl = detect_anomalies(df_googl)
print("🔍 Detected anomalies in Google (GOOGL):\n", df_googl[df_googl['anomaly']])
plot_anomalies(df_googl, ticker_googl)
df_googl[df_googl['anomaly']].to_csv("GOOGL_anomalies.csv")
