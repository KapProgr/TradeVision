# 📈 MarketMind AI: Intelligent Stock Forecasting System

A professional-grade financial analysis tool built with **Python** and **Streamlit**. It leverages **Deep Learning (LSTM Neural Networks)** to analyze historical market data, calculate technical indicators, and predict future price movements with high precision.

This project goes beyond simple line-fitting by implementing a **"Leakage-Free"** data pipeline, ensuring that the model is trained strictly on past data without "peeking" into the future (a common pitfall in financial AI).

## 🚀 Live Demo
**[Click here to view the App](ΤΟ_LINK_ΤΟΥ_STREAMLIT_ΣΟΥ_ΕΔΩ)**

## ✨ Key Features

* **🧠 Deep Learning Core (LSTM):** Powered by **TensorFlow/Keras** Long Short-Term Memory networks, designed specifically to capture temporal dependencies and patterns in time-series data.
* **🛡️ Leakage-Free Architecture:** Implements a robust preprocessing pipeline where `MinMaxScaler` fits *only* on training data, preventing data leakage and ensuring realistic performance metrics.
* **📊 Advanced Technical Analysis:** Automatically computes key financial indicators including **RSI, MACD, Bollinger Bands, and Moving Averages (MA7/21/50)**.
* **🌍 Universal Search:** Fetches real-time data for **Stocks (US/EU/Greek), Crypto, and Forex** using the Yahoo Finance API (supports custom tickers like `BTC-USD`, `NVDA`, `EURUSD=X`).
* **📉 Interactive Visualization:** Features professional-grade **Plotly** charts (Candlestick, Forecasts) with zoom/pan capabilities.
* **⚡ Smart Timeframes:** Pre-configured modes for rapid analysis ("Fast 3-Year" vs "Deep History") to balance training speed and data depth.

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit** (Frontend/UI)
* **TensorFlow / Keras** (LSTM Model Construction)
* **yFinance** (Real-time Market Data)
* **Scikit-Learn** (Preprocessing & Metrics)
* **Plotly** (Financial Visualization)
* **Pandas & NumPy** (Time-series Manipulation)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KapProgr/marketmind-ai.git](https://github.com/KapProgr/marketmind-ai.git)
    cd marketmind-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    python -m streamlit run stock_app.py
    ```

## 🧠 Model Architecture
The AI model is built using a sequential architecture optimized for time-series forecasting:
1.  **Input Layer:** Processes sequences of historical data (Open, Close, Volume, Indicators).
2.  **LSTM Layers:** Two stacked LSTM layers to learn complex temporal patterns.
3.  **Dropout Layers:** Applied (default 0.2) to prevent overfitting.
4.  **Dense Output:** Predicts the next day's closing price.

## 📂 Dataset
Data is fetched dynamically via the **Yahoo Finance API**. No local CSV files are required. The app supports:
* **US Tech Giants** (Apple, Tesla, NVIDIA)
* **Indices & ETFs** (S&P 500, Gold, Bitcoin)
* **International Markets** (Greek Stock Exchange, European stocks)

## ⚠️ Disclaimer
This application is for **educational and research purposes only**. It uses historical data to generate predictions and should not be used as the sole basis for real-money investment decisions.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests.

## 📜 License
This project is open-source.