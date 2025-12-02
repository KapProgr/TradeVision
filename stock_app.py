import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        background-color: #0e1117;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464646;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b8b8b;
    }
    h1 {
        color: #00d4ff;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff 0%, #0066ff 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0066ff 0%, #00d4ff 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Popular stock tickers with IPO dates
POPULAR_STOCKS = {
    "🇺🇸 US Tech Giants": {
        "Apple": {"ticker": "AAPL", "ipo": "1980-12-12"},
        "Microsoft": {"ticker": "MSFT", "ipo": "1986-03-13"},
        "Google": {"ticker": "GOOGL", "ipo": "2004-08-19"},
        "Amazon": {"ticker": "AMZN", "ipo": "1997-05-15"},
        "Meta (Facebook)": {"ticker": "META", "ipo": "2012-05-18"},
        "Tesla": {"ticker": "TSLA", "ipo": "2010-06-29"},
        "NVIDIA": {"ticker": "NVDA", "ipo": "1999-01-22"},
        "Netflix": {"ticker": "NFLX", "ipo": "2002-05-23"},
        "Adobe": {"ticker": "ADBE", "ipo": "1986-08-20"},
        "Intel": {"ticker": "INTC", "ipo": "1971-10-13"},
        "AMD": {"ticker": "AMD", "ipo": "1979-09-01"},
        "Oracle": {"ticker": "ORCL", "ipo": "1986-03-12"},
        "Salesforce": {"ticker": "CRM", "ipo": "2004-06-23"},
        "PayPal": {"ticker": "PYPL", "ipo": "2002-02-15"},
        "Uber": {"ticker": "UBER", "ipo": "2019-05-10"},
        "Airbnb": {"ticker": "ABNB", "ipo": "2020-12-10"},
        "Spotify": {"ticker": "SPOT", "ipo": "2018-04-03"},
        "Zoom": {"ticker": "ZM", "ipo": "2019-04-18"}
    },
    "🇺🇸 US Finance": {
        "JPMorgan Chase": {"ticker": "JPM", "ipo": "1969-03-05"},
        "Bank of America": {"ticker": "BAC", "ipo": "1972-01-01"},
        "Goldman Sachs": {"ticker": "GS", "ipo": "1999-05-04"},
        "Wells Fargo": {"ticker": "WFC", "ipo": "1972-01-01"},
        "Morgan Stanley": {"ticker": "MS", "ipo": "1986-03-05"},
        "Citigroup": {"ticker": "C", "ipo": "1977-01-01"},
        "Visa": {"ticker": "V", "ipo": "2008-03-19"},
        "Mastercard": {"ticker": "MA", "ipo": "2006-05-25"},
        "American Express": {"ticker": "AXP", "ipo": "1977-05-18"},
        "BlackRock": {"ticker": "BLK", "ipo": "1999-10-01"}
    },
    "🇺🇸 US Consumer": {
        "Coca-Cola": {"ticker": "KO", "ipo": "1919-09-05"},
        "Pepsi": {"ticker": "PEP", "ipo": "1972-06-08"},
        "McDonald's": {"ticker": "MCD", "ipo": "1965-04-21"},
        "Starbucks": {"ticker": "SBUX", "ipo": "1992-06-26"},
        "Nike": {"ticker": "NKE", "ipo": "1980-12-02"},
        "Disney": {"ticker": "DIS", "ipo": "1957-11-12"},
        "Walmart": {"ticker": "WMT", "ipo": "1972-10-01"},
        "Target": {"ticker": "TGT", "ipo": "1967-10-18"},
        "Costco": {"ticker": "COST", "ipo": "1985-12-05"},
        "Home Depot": {"ticker": "HD", "ipo": "1981-09-22"}
    },
    "🇺🇸 US Healthcare": {
        "Johnson & Johnson": {"ticker": "JNJ", "ipo": "1944-09-25"},
        "Pfizer": {"ticker": "PFE", "ipo": "1972-06-22"},
        "UnitedHealth": {"ticker": "UNH", "ipo": "1984-10-18"},
        "Abbott Labs": {"ticker": "ABT", "ipo": "1929-03-18"},
        "Merck": {"ticker": "MRK", "ipo": "1941-01-01"},
        "Moderna": {"ticker": "MRNA", "ipo": "2018-12-07"},
        "CVS Health": {"ticker": "CVS", "ipo": "1996-03-01"}
    },
    "🇺🇸 US Energy": {
        "ExxonMobil": {"ticker": "XOM", "ipo": "1970-01-01"},
        "Chevron": {"ticker": "CVX", "ipo": "1970-01-01"},
        "ConocoPhillips": {"ticker": "COP", "ipo": "1929-01-01"},
        "NextEra Energy": {"ticker": "NEE", "ipo": "1984-05-01"}
    },
    "🇬🇷 Greek Stocks": {
        "ΟΠΑΠ": {"ticker": "OPAP.AT", "ipo": "2001-05-31"},
        "Alpha Bank": {"ticker": "ALPHA.AT", "ipo": "1991-11-13"},
        "Εθνική Τράπεζα": {"ticker": "ETE.AT", "ipo": "1880-01-01"},
        "ΔΕΗ": {"ticker": "PPC.AT", "ipo": "2001-12-14"},
        "Eurobank": {"ticker": "EUROB.AT", "ipo": "1999-10-21"},
        "Μυτιληναίος": {"ticker": "MYTIL.AT", "ipo": "1995-12-20"},
        "ΟΤΕ": {"ticker": "OTE.AT", "ipo": "1996-12-19"},
        "Τιτάν": {"ticker": "TITC.AT", "ipo": "1912-01-01"},
        "Jumbo": {"ticker": "BELA.AT", "ipo": "1999-07-06"},
        "Ελληνικά Πετρέλαια": {"ticker": "ELPE.AT", "ipo": "1998-06-30"},
        "Motor Oil": {"ticker": "MOH.AT", "ipo": "2001-07-20"},
        "Coca-Cola HBC": {"ticker": "EEEK.AT", "ipo": "2000-08-02"}
    },
    "💰 Crypto & Commodities": {
        "Bitcoin ETF": {"ticker": "BITO", "ipo": "2021-10-19"},
        "Ethereum ETF": {"ticker": "EETH", "ipo": "2023-10-02"},
        "Gold ETF (GLD)": {"ticker": "GLD", "ipo": "2004-11-18"},
        "Silver ETF (SLV)": {"ticker": "SLV", "ipo": "2006-04-28"}
    },
    "🌍 International": {
        "Alibaba (China)": {"ticker": "BABA", "ipo": "2014-09-19"},
        "Samsung (Korea)": {"ticker": "005930.KS", "ipo": "1975-06-11"},
        "Toyota (Japan)": {"ticker": "TM", "ipo": "1949-05-16"},
        "BP (UK)": {"ticker": "BP", "ipo": "1954-01-01"},
        "Nestlé (Switzerland)": {"ticker": "NSRGY", "ipo": "1905-01-01"},
        "SAP (Germany)": {"ticker": "SAP", "ipo": "1988-11-04"}
    }
}

class StockPredictorUI:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    @st.cache_data(ttl="1h", show_spinner="Fetching stock data...")
    def fetch_data(_self, ticker, start_date, end_date):
        """Fetch stock data"""
        try:
            # Convert dates to string format
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            
            data = yf.download(ticker, start=start_str, end=end_str, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data if len(data) > 0 else None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    @st.cache_data(ttl="1h", show_spinner="Calculating technical indicators...")
    def add_technical_indicators(_self, df):
        """Add technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        
        df.dropna(inplace=True)
        return df

    def prepare_data(self, df, features, lookback, train_split):
        """
        Correct Data Preparation to avoid Data Leakage.
        Scales based only on training data.
        """
        # 1. Split data into training and testing sets
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # 2. Fit the scaler ONLY on the training data to prevent data leakage
        train_data = train_df[features].values
        self.scaler.fit(train_data)

        # 3. Scale both training and testing data using the fitted scaler
        scaled_train = self.scaler.transform(train_data)
        scaled_test_data = self.scaler.transform(test_df[features].values)

        # Combine last 'lookback' days of training data with test data for sequence creation
        last_train_sequence = scaled_train[-lookback:]
        scaled_test = np.concatenate([last_train_sequence, scaled_test_data])

        def create_sequences(dataset, lookback):
            X, y = [], []
            for i in range(lookback, len(dataset)):
                X.append(dataset[i-lookback:i])
                y.append(dataset[i, 0]) # Target is the first feature ('Close' price)
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(scaled_train, lookback)
        X_test, y_test = create_sequences(scaled_test, lookback)
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape, units_1, units_2, units_3, dropout):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=units_1, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units=units_2, return_sequences=True),
            Dropout(dropout),
            LSTM(units=units_3, return_sequences=False),
            Dropout(dropout),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @st.cache_resource(show_spinner="Training AI model... (this may take a few minutes)")
    def train_model(_self, X_train, y_train, X_test, y_test, epochs, batch_size, 
                   units_1, units_2, units_3, dropout):
        """Train LSTM model"""
        model = _self.build_lstm_model(
            (X_train.shape[1], X_train.shape[2]),
            units_1, units_2, units_3, dropout
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        return model, history
    
    def inverse_transform_predictions(_self, predictions, num_features):
        """Inverse transform scaled predictions"""
        dummy = np.zeros((len(predictions), num_features))
        dummy[:, 0] = predictions.flatten()
        return _self.scaler.inverse_transform(dummy)[:, 0]
    
    def predict_future(self, model, df, features, lookback, days):
        """Predict future prices"""
        last_sequence = df[features].values[-lookback:]
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        future_predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(days):
            current_batch = current_sequence.reshape(1, lookback, len(features))
            next_pred = model.predict(current_batch, verbose=0)
            
            next_step = current_sequence[-1].copy()
            next_step[0] = next_pred[0, 0]
            
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.vstack([current_sequence[1:], next_step])
        
        return self.inverse_transform_predictions(np.array(future_predictions), len(features))

def plot_stock_data(df, ticker):
    """Plot historical stock data with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and MAs
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='MA7',
                            line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA21'], name='MA21',
                            line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                            line=dict(color='purple', width=1)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='cyan', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='blue', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal',
                            line=dict(color='red', width=2)), row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} - Technical Analysis',
        height=900,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def plot_predictions(df, predictions_actual, y_test_actual, ticker):
    test_dates = df.index[-len(y_test_actual):]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_test_actual,
        mode='lines', name='Actual Price',
        line=dict(color='#00d4ff', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, y=predictions_actual,
        mode='lines', name='Predicted Price',
        line=dict(color='#ff6b6b', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{ticker} - Predictions vs Actual',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_future_forecast(df, future_prices, future_dates, ticker):
    historical_dates = df.index[-90:]
    historical_prices = df['Close'].values[-90:]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_prices,
        mode='lines', name='Historical',
        line=dict(color='#00d4ff', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode='lines+markers', name='Forecast',
        line=dict(color='#00ff88', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'{ticker} - Future Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def get_currency_symbol(ticker):
    if ticker.endswith(('.AT', '.DE', '.PA')): 
        return "€"
    if ticker.endswith('.L'): 
        return "£"
    return "$"

# Main App
def main():
    st.markdown("<h1>📈 AI Stock Price Predictor</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Ρυθμίσεις Ανάλυσης")
        
        # Stock Selection
        st.subheader("📊 Επιλογή Μετοχής")
        
        custom_ticker = st.text_input("Ή πληκτρολογήστε σύμβολο (ticker):", placeholder="π.χ. GOOGL, MSFT")
        use_custom_ticker = bool(custom_ticker)

        category = st.selectbox("Κατηγορία", list(POPULAR_STOCKS.keys()), disabled=use_custom_ticker)
        stock_name = st.selectbox("Μετοχή", list(POPULAR_STOCKS[category].keys()), disabled=use_custom_ticker)
        
        if custom_ticker:
            ticker = custom_ticker.upper()
            ipo_date = "1970-01-01"  
            st.info(f"**Επιλεγμένο Σύμβολο:** {ticker}")
        else:
            selected_stock = POPULAR_STOCKS[category][stock_name]
            ticker = selected_stock["ticker"]
            ipo_date = selected_stock["ipo"]
            st.info(f"**{stock_name}** ({ticker})\n\n📅 IPO: {ipo_date}")
        
        currency_symbol = get_currency_symbol(ticker)
        
        st.markdown("---")
        
        # Date Range
        st.subheader("📅 Χρονικό Διάστημα")
        
        # Preset options
        preset = st.radio(
            "Επιλέξτε περίοδο δεδομένων:",
            ["⚡ Γρήγορο (3 έτη - Προτείνεται)", "⚖️ Ισορροπημένο (5 έτη)", "🎯 Μέγιστο (Πλήρες ιστορικό)", "🔧 Προσαρμοσμένο"],
            help="Τα 3 έτη είναι ιδανικά για τις περισσότερες μετοχές - γρήγορη εκπαίδευση με καλή ακρίβεια."
        )
        
        if preset == "⚡ Γρήγορο (3 έτη - Προτείνεται)":
            start_date = pd.to_datetime("today") - pd.DateOffset(years=3)
            end_date = pd.to_datetime("today")
            st.success("📊 Χρήση δεδομένων **3 ετών** (~750 ημέρες) ⚡ Εκπαίδευση: ~30-45 δευτερόλεπτα")
            
        elif preset == "⚖️ Ισορροπημένο (5 έτη)":
            start_date = pd.to_datetime("today") - pd.DateOffset(years=5)
            end_date = pd.to_datetime("today")
            st.info("📊 Χρήση δεδομένων **5 ετών** (~1250 ημέρες) ⚖️ Εκπαίδευση: ~1-2 λεπτά")
            
        elif preset == "🎯 Μέγιστο (Πλήρες ιστορικό)":
            try:
                start_date_default = pd.to_datetime(ipo_date)
                start_date_default = start_date_default + pd.DateOffset(years=1)
            except:
                start_date_default = pd.to_datetime("2015-01-01")
            
            start_date = start_date_default
            end_date = pd.to_datetime("today")
            
            days_diff = (end_date - start_date).days
            
            if days_diff > 2000:
                st.warning(f"⚠️ Μεγάλο σύνολο δεδομένων: **{days_diff} ημέρες** (~{days_diff//252} έτη). Εκπαίδευση: 3-5+ λεπτά")
            else:
                st.info(f"📊 Χρήση **πλήρους ιστορικού**: {days_diff} ημέρες")
                
        else:  # Custom
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Ημερομηνία Έναρξης", pd.to_datetime("2022-01-01"))
            with col2:
                end_date = st.date_input("Ημερομηνία Λήξης", pd.to_datetime("today"))
        
        st.markdown("---")
        
        # Model Parameters
        st.subheader("🧠 Παράμετροι Μοντέλου")
        
        lookback = st.slider("Lookback Days", 20, 120, 60, 10,
                            help="Number of past days to consider")
        
        train_split = st.slider("Train Split %", 60, 95, 80, 5,
                               help="Percentage of data for training") / 100
        
        epochs = st.slider("Training Epochs", 10, 200, 30, 10,
                          help="Number of training iterations (Lower = Faster, Higher = Potentially more accurate)")
        
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        
        st.markdown("---")
        
        # LSTM Architecture
        st.subheader("🏗️ Αρχιτεκτονική LSTM")
        
        units_1 = st.slider("Layer 1 Units", 50, 200, 100, 25)
        units_2 = st.slider("Layer 2 Units", 50, 200, 100, 25)
        units_3 = st.slider("Layer 3 Units", 25, 100, 50, 25)
        dropout = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
        
        st.markdown("---")
        
        # Prediction
        st.subheader("🔮 Μελλοντική Πρόβλεψη")
        prediction_days = st.slider("Days to Predict", 7, 90, 30, 7)
        
        st.markdown("---")
        
        # Features Selection
        st.subheader("📊 Features για Ανάλυση")
        all_features = ['Close', 'Volume', 'MA7', 'MA21', 'MA50', 'RSI', 'MACD']
        features = st.multiselect(
            "Select Features",
            all_features,
            default=['Close', 'Volume', 'MA7', 'MA21', 'RSI', 'MACD']
        )
        
        if len(features) == 0:
            st.error("Select at least one feature!")
            return
        
        st.markdown("---")
        
        run_button = st.button("🚀 Run Analysis", use_container_width=True)
    
    # Main Content
    if run_button:
        with st.spinner(f"Starting analysis for {ticker}..."):
             predictor = StockPredictorUI()
             data = predictor.fetch_data(ticker, start_date, end_date)

        if data is None or len(data) < 100:
            st.error("❌ Not enough data! Try different dates or stock.")
            return
        
        st.success(f"✅ Fetched {len(data)} days of data for {ticker}")
        
        # --- Start of Processing ---
        status_placeholder = st.empty()
        with status_placeholder.container():
            data = predictor.add_technical_indicators(data)

            with st.spinner("Preparing data for the model..."):
                X_train, X_test, y_train, y_test = predictor.prepare_data(
                    data, features, lookback, train_split
                )
                
            # This will now use the cache if parameters are the same
            model, history = predictor.train_model(
                    X_train, y_train, X_test, y_test,
                    epochs, batch_size, units_1, units_2, units_3, dropout
                )

            with st.spinner("Generating predictions and forecasts..."):
                predictions = model.predict(X_test, verbose=0)
                predictions_actual = predictor.inverse_transform_predictions(predictions, len(features))
                y_test_actual = predictor.inverse_transform_predictions(y_test, len(features))

                future_prices = predictor.predict_future(model, data, features, lookback, prediction_days)
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date, periods=prediction_days + 1)[1:]
        
        status_placeholder.success("✅ Analysis Complete!")

        # --- Display Results in Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Analysis", "🎯 Model Performance", "📈 Predictions vs Actual", "🔮 Future Forecast"])

        with tab1:
            st.subheader(f"Current Metrics for {ticker}")
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Current Price", f"{currency_symbol}{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            c2.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            c3.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
            c4.metric("High (1Y)", f"{currency_symbol}{data['High'].tail(252).max():.2f}")
            c5.metric("Low (1Y)", f"{currency_symbol}{data['Low'].tail(252).min():.2f}")

            st.subheader("Historical Data & Indicators")
            fig_tech = plot_stock_data(data, ticker)
            st.plotly_chart(fig_tech, use_container_width=True)

        with tab2:
            st.subheader("Model Performance Metrics")
            mse = mean_squared_error(y_test_actual, predictions_actual)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, predictions_actual)
            r2 = r2_score(y_test_actual, predictions_actual)
            mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{currency_symbol}{rmse:.2f}", help="Root Mean Squared Error. Lower is better.")
            c2.metric("MAE", f"{currency_symbol}{mae:.2f}", help="Mean Absolute Error. Lower is better.")
            c3.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination. Closer to 1 is better. Represents the proportion of the variance for a dependent variable that's explained by an independent variable.")
            c4.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error. Lower is better.")

            st.subheader("Training History")
            c1, c2 = st.columns(2)
            with c1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='#00d4ff', width=2)))
                fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', line=dict(color='#ff6b6b', width=2)))
                fig_loss.update_layout(title='Training & Validation Loss', template='plotly_dark', height=350, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig_loss, use_container_width=True)
            with c2:
                accuracy_pct = (1 - mape/100) * 100
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=accuracy_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Accuracy (based on MAPE)"},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#00d4ff"},
                           'steps': [{'range': [0, 80], 'color': "#3d3d4d"}, {'range': [80, 100], 'color': "#464659"}]}
                ))
                fig_gauge.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig_gauge, use_container_width=True)

        with tab3:
            st.subheader("Model Predictions vs. Actual Prices")
            fig_pred = plot_predictions(data, predictions_actual, y_test_actual, ticker)
            st.plotly_chart(fig_pred, use_container_width=True)

        with tab4:
            st.subheader(f"Forecast for the Next {prediction_days} Days")
            fig_future = plot_future_forecast(data, future_prices, future_dates, ticker)
            st.plotly_chart(fig_future, use_container_width=True)

            st.subheader("Forecast Summary")
            price_change = future_prices[-1] - current_price
            pct_change = (price_change / current_price) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
            c2.metric(f"Predicted ({prediction_days}d)", f"{currency_symbol}{future_prices[-1]:.2f}")
            c3.metric("Expected Change", f"{currency_symbol}{price_change:+.2f}")
            c4.metric("Expected % Change", f"{pct_change:+.2f}%")

            if pct_change > 5:
                st.success(f"🟢 **Bullish Signal**: The model predicts a significant increase of {pct_change:.2f}% over the next {prediction_days} days.")
            elif pct_change < -5:
                st.error(f"🔴 **Bearish Signal**: The model predicts a significant decrease of {pct_change:.2f}% over the next {prediction_days} days.")
            else:
                st.info(f"🟡 **Neutral Signal**: The model predicts a relatively stable change of {pct_change:.2f}% over the next {prediction_days} days.")

            st.subheader("Detailed Forecast Data")
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Predicted Price': [f"{currency_symbol}{p:.2f}" for p in future_prices],
                'Change from Today': [f"{((p-current_price)/current_price*100):+.2f}%" for p in future_prices]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            st.warning("⚠️ **Disclaimer**: This is an AI prediction tool for educational purposes. Always conduct your own research and consult financial advisors before making investment decisions.")

if __name__ == "__main__":
    main()