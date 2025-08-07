import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import ccxt
import time
import os

# Configurações iniciais
st.set_page_config(page_title="IA Trader Bot", layout="wide")
st.title("🤖 Bot de Trading com IA - BTC/ETH")

# Sidebar
st.sidebar.header("Configurações do Bot")
par = st.sidebar.selectbox("Escolha o par:", ["BTC/USDT", "ETH/USDT"])
modo_simulacao = st.sidebar.checkbox("Modo Simulação", value=True)
freq_segundos = st.sidebar.number_input("Frequência de atualização (segundos)", min_value=10, max_value=3600, value=60, step=10)
executar = st.sidebar.toggle("Ativar bot 24/7", value=False)

# API Keys
api_key = st.sidebar.text_input("Binance API Key", type="password")
api_secret = st.sidebar.text_input("Binance API Secret", type="password")

placeholder = st.empty()

# Função para baixar e preparar dados
def baixar_dados(symbol, look_back=10):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    data = df[['close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return df, X, y, scaler

# Função para criar e treinar modelo
def treinar_modelo(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

# Função para executar ordem de compra/venda
def executar_ordem(exchange, symbol, side, quantidade):
    try:
        order = exchange.create_market_order(symbol=symbol, side=side, amount=quantidade)
        return order
    except Exception as e:
        return str(e)

# Loop de execução
while executar:
    with placeholder.container():
        st.subheader(f"Analisando {par} | Modo: {'Simulação' if modo_simulacao else 'Real'}")
        df, X, y, scaler = baixar_dados(par)
        model = treinar_modelo(X, y)

        predicted = model.predict(X)
        predicted = scaler.inverse_transform(predicted)
        real = scaler.inverse_transform(y.reshape(-1, 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(real, label='Real')
        ax.plot(predicted, label='Previsto')
        ax.set_title("Preço previsto vs. real")
        ax.legend()
        st.pyplot(fig)

        preco_atual = df['close'].iloc[-1]
        previsao = predicted[-1][0]

        st.write(f"Preço atual: {preco_atual:.2f} | Previsão: {previsao:.2f}")

        if not modo_simulacao and api_key and api_secret:
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })

            if previsao > preco_atual * 1.002:
                resultado = executar_ordem(exchange, par, 'buy', 0.001)
                st.info(f"Ordem de COMPRA executada: {resultado}")
            elif previsao < preco_atual * 0.998:
                resultado = executar_ordem(exchange, par, 'sell', 0.001)
                st.info(f"Ordem de VENDA executada: {resultado}")
        elif modo_simulacao:
            if previsao > preco_atual * 1.002:
                st.info("Simulação: Ordem de COMPRA sugerida")
            elif previsao < preco_atual * 0.998:
                st.info("Simulação: Ordem de VENDA sugerida")

        st.success(f"Atualizado em: {time.strftime('%H:%M:%S')}")

    time.sleep(freq_segundos)

st.warning("Bot parado. Ative o toggle na barra lateral para iniciar.")