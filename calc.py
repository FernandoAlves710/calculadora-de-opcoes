import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go

# Função para obter os dados do ativo (ação ou ETF) do Yahoo Finance
def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatility = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatility, hist

# Função para calcular o preço da opção usando o modelo de Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Função para calcular o preço da opção americana usando o método de Monte Carlo
def monte_carlo_option_pricing_american(S, K, T, r, sigma, num_simulations=10000):
    dt = T / 365
    Z = np.random.normal(0, 1, (num_simulations, int(T * 365)))
    ST = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1))
    payoff = np.maximum(ST - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff, axis=1)
    return np.mean(option_price)

# Função para calcular o preço da opção asiática usando o método de Monte Carlo
def monte_carlo_option_pricing_asian(S, K, T, r, sigma, num_simulations=10000):
    dt = T / 365
    Z = np.random.normal(0, 1, (num_simulations, int(T * 365)))
    ST = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1))
    ST_mean = np.mean(ST, axis=1)
    payoff = np.maximum(ST_mean - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# Interface do usuário
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

# Estilos personalizados
st.markdown("""
<style>
    .big-font {
        font-size:25px !important;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #0e1117;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #036;
    }
    .description {
        font-size: 16px;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title('Calculadora de Opções Avançada')

# Sidebar para entrada de dados
st.write("## Parâmetros de Entrada")
simbolo = st.text_input("Digite o símbolo do ativo desde moedas, ações, commodities e etfs (ex: AAPL):")

if simbolo:
    S, volatility, hist = get_stock_data(simbolo)
    st.write(f"Preço Atual do Ativo: {S:.2f}")
    st.write(f"Volatilidade Anualizada: {volatility:.2%}")

    K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=S, format="%.2f")
    T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    option_type = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

    if option_type == "Europeia":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = black_scholes(S, K, T, r, volatility)
            vol_imp = implied_volatility(S, K, T, r, preco_opcao)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write("### Gregas:")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f} (sensibilidade do preço da opção em relação ao preço do ativo subjacente)")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f} (sensibilidade do delta em relação ao preço do ativo subjacente)")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f} (sensibilidade do preço da opção em relação à volatilidade do ativo subjacente)")
            st.write("### Volatilidade Implícita:")
            st.write(f"{vol_imp:.2%}", cls="result")
            st.write("Descrição: Volatilidade implícita é a volatilidade futura do ativo subjacente, inferida do preço atual da opção.")
    
    elif option_type == "Americana":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = monte_carlo_option_pricing_american(S, K, T, r, volatility)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write("### Gregas:")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f} (sensibilidade do preço da opção em relação ao preço do ativo subjacente)")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f} (sensibilidade do delta em relação ao preço do ativo subjacente)")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f} (sensibilidade do preço da opção em relação à volatilidade do ativo subjacente)")
            st.write("### Volatilidade Implícita:")
            st.write(f"{vol_imp:.2%}", cls="result")
            st.write("Descrição: Volatilidade implícita é a volatilidade futura do ativo subjacente, inferida do preço atual da opção.")

    elif option_type == "Asiática":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = monte_carlo_option_pricing_asian(S, K, T, r, volatility)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write("### Gregas:")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f} (sensibilidade do preço da opção em relação ao preço do ativo subjacente)")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f} (sensibilidade do delta em relação ao preço do ativo subjacente)")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f} (sensibilidade do preço da opção em relação à volatilidade do ativo subjacente)")
            st.write("### Volatilidade Implícita:")
            st.write(f"{vol_imp:.2%}", cls="result")
            st.write("Descrição: Volatilidade implícita é a volatilidade futura do ativo subjacente, inferida do preço atual da opção.")

    st.write("## Histórico de Preços")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Histórico de Preços do Ativo nos Últimos 12 Meses',
                      xaxis_title='Data', yaxis_title='Preço')
    st.plotly_chart(fig)
