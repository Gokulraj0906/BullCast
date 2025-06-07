import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

stock_list = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "HDFCBANK.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "WIPRO.NS", "SUNPHARMA.NS", "POWERGRID.NS",
    "ONGC.NS", "NTPC.NS", "TECHM.NS", "TITAN.NS", "TATASTEEL.NS", "COALINDIA.NS", "HCLTECH.NS",
    "GRASIM.NS", "ADANIENT.NS", "ADANIPORTS.NS", "UPL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "JSWSTEEL.NS", "EICHERMOT.NS", "HINDALCO.NS", "DRREDDY.NS", "DIVISLAB.NS", "NESTLEIND.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "M&M.NS", "BAJAJFINSV.NS", "SHREECEM.NS", "SBILIFE.NS",
    "ICICIPRULI.NS", "HDFCLIFE.NS", "HAVELLS.NS", "PIDILITIND.NS", "GODREJCP.NS", "DMART.NS",
    "DABUR.NS", "TATAMOTORS.NS", "INDUSINDBK.NS", "TORNTPHARM.NS", "HDFCAMC.NS", "LUPIN.NS",
    "SRF.NS", "AMBUJACEM.NS", "TATAPOWER.NS", "BEL.NS", "BOSCHLTD.NS", "ABB.NS", "AUBANK.NS",
    "APOLLOHOSP.NS", "PNB.NS", "INDIGO.NS", "CANBK.NS", "BANKBARODA.NS", "RECLTD.NS",
    "PEL.NS", "AUROPHARMA.NS", "CHOLAFIN.NS", "BANDHANBNK.NS", "ZEEL.NS", "BIOCON.NS",
    "IDFCFIRSTB.NS", "INDHOTEL.NS", "ESCORTS.NS", "TVSMOTOR.NS", "VOLTAS.NS", "PAGEIND.NS",
    "COLPAL.NS", "BERGEPAINT.NS", "ICICIGI.NS", "LICHSGFIN.NS", "MUTHOOTFIN.NS", "GLAND.NS",
    "IRCTC.NS", "METROPOLIS.NS", "NAUKRI.NS", "IEX.NS", "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS",
    "MAPMYINDIA.NS", "TRIDENT.NS", "DEEPAKNTR.NS", "RAIN.NS", "MCX.NS", "BATAINDIA.NS",
    "ALKEM.NS", "CROMPTON.NS", "ICRA.NS", "LTI.NS", "LTTS.NS", "MINDTREE.NS", "MPHASIS.NS",
    "POLYCAB.NS", "RAJESHEXPO.NS", "RBLBANK.NS", "SAIL.NS", "SYNGENE.NS", "TATACOMM.NS",
    "TRENT.NS", "UBL.NS", "UNIONBANK.NS", "UCOBANK.NS", "IDBI.NS", "FEDERALBNK.NS",
    "IIFL.NS", "ABFRL.NS", "CONCOR.NS", "INDIAMART.NS", "BALRAMCHIN.NS", "PIIND.NS",
    "PERSISTENT.NS", "JUBLFOOD.NS", "NAM-INDIA.NS", "NHPC.NS", "IRFC.NS", "PVRINOX.NS",
    "TATAELXSI.NS", "COFORGE.NS", "BHARATFORG.NS", "HINDPETRO.NS", "GAIL.NS", "IOC.NS",
    "GMRINFRA.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "ADANIPOWER.NS", "IGL.NS", "MGL.NS"
]

def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error("Error fetching stock data.")
        st.exception(e)
        return pd.DataFrame()

def add_technical_indicators(df):
    try:
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error("Error calculating technical indicators.")
        st.exception(e)
        return df

def create_features(df):
    try:
        df['Lag1'] = df['Close'].shift(1)
        df['Lag2'] = df['Close'].shift(2)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error("Error creating lag features.")
        st.exception(e)
        return df

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_signal(actual, predicted):
    return "📈 BUY" if predicted[-1] > actual[-1] else "📉 SELL"


st.set_page_config(page_title="Stock Predictor", layout="wide", initial_sidebar_state="expanded")
st.title(" NSE/BSE Stock Market Predictor")
st.markdown("Use AI models to predict future stock prices and generate Buy/Sell signals.")

ticker = st.selectbox("🔍 Select NSE/BSE Stock", stock_list)
model_option = st.selectbox("🧠 Choose Prediction Model", ["LSTM", "Random Forest", "XGBoost"])

start_date = st.date_input("📅 Start Date", datetime.now() - timedelta(days=365*2))
end_date = st.date_input("📅 End Date", datetime.now())

if st.button("🚀 Predict"):
    st.subheader(f"Stock Analysis: {ticker}")
    data = fetch_stock_data(ticker, start_date, end_date)

    if not data.empty:
        data = add_technical_indicators(data)
        data = create_features(data)

        X = data[['Lag1', 'Lag2', 'RSI', 'SMA_10', 'SMA_50']]
        y = data['Close']

        try:
            if model_option == "Random Forest":
                model = RandomForestRegressor()
                model.fit(X, y)
                pred = model.predict([X.iloc[-1]])

            elif model_option == "XGBoost":
                model = XGBRegressor(objective='reg:squarederror')
                model.fit(X, y)
                pred = model.predict([X.iloc[-1]])

            else: 
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

                X_seq = [X_scaled[i-2:i] for i in range(2, len(X_scaled))]
                X_seq = np.array(X_seq)
                y_seq = y_scaled[2:]

                model = create_lstm_model((X_seq.shape[1], X_seq.shape[2]))
                model.fit(X_seq, y_seq, epochs=10, batch_size=8, verbose=0)

                last_input = np.array([X_scaled[-2:]])
                pred_scaled = model.predict(last_input)
                pred = scaler.inverse_transform(pred_scaled)[0]

        except Exception as e:
            st.error("Model training/prediction error.")
            st.exception(e)
            pred = [0]

        signal = generate_signal(data['Close'].values, pred)

        st.metric(label="💹 Predicted Price", value=f"₹{pred[0]:.2f}")
        st.metric(label="📢 Buy/Sell Signal", value=signal)

        st.write("### Price Chart with Indicators")
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Close'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], mode='lines', name='SMA 10'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))

            fig.update_layout(
                title=f"{ticker} - Price Chart with Indicators",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                legend=dict(x=0, y=1),
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Error generating line chart.")
            st.exception(e)

        st.write("### 📋 Recent Data Snapshot")
        st.dataframe(data.tail(10))

        st.write("### 📈 Technical Indicators")
        st.dataframe(data[['Close', 'SMA_10', 'SMA_50', 'RSI']].tail(10))

        st.markdown("---")
        st.markdown("""
        **🔒 Disclaimer:**  
        This application is for **educational purposes only**. Predictions are not financial advice.  
        Consult a licensed financial advisor before making investment decisions.
        """)