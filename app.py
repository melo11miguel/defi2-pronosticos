import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta

# =========================================================
# DESCARGA DE DATOS (ROBUSTA)
# =========================================================
def download_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        return None

    for col in ["Adj Close", "Close", "close", "adjclose"]:
        if col in data.columns:
            return data[col].dropna().astype(float)

    return None


# =========================================================
# AUXILIARES
# =========================================================
def train_test_split(series, test_size=0.2):
    n = len(series)
    n_test = int(n * test_size)
    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]
    return train, test


def compute_rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# =========================================================
# MODELO 1: GBM
# =========================================================
def backtest_gbm(prices_train, prices_test, n_paths=1000, seed=42):

    train_arr = prices_train.values.astype(float)
    test_arr = prices_test.values.astype(float)

    log_ret = np.log(train_arr[1:] / train_arr[:-1])
    mu = log_ret.mean() * 252
    sigma = log_ret.std() * np.sqrt(252)
    dt = 1 / 252

    S0 = float(train_arr[-1])
    n_steps = len(test_arr)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))

    S_paths = np.zeros((n_paths, n_steps))
    S_paths[:, 0] = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, 0])

    for t in range(1, n_steps):
        S_paths[:, t] = (
            S_paths[:, t-1] *
            np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])
        )

    preds = S_paths.mean(axis=0)
    return preds, compute_rmse(test_arr, preds), {"mu": mu, "sigma": sigma}


# =========================================================
# MODELO 2: HESTON
# =========================================================
def backtest_heston(prices_train, prices_test,
                    kappa=2.0, theta_factor=1.0, sigma_v=0.5, rho=-0.7,
                    n_paths=1000, seed=123):

    train_arr = prices_train.values.astype(float)
    test_arr = prices_test.values.astype(float)

    log_ret = np.log(train_arr[1:] / train_arr[:-1])
    mu = log_ret.mean() * 252
    v0 = np.var(log_ret) * 252
    theta = theta_factor * v0
    dt = 1 / 252

    S0 = float(train_arr[-1])
    n_steps = len(test_arr)

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2i = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2i

    S_paths = np.zeros((n_paths, n_steps))
    v_paths = np.zeros((n_paths, n_steps))

    S_paths[:, 0] = S0
    v_paths[:, 0] = max(v0, 1e-8)

    for t in range(1, n_steps):
        v_prev = np.maximum(v_paths[:, t-1], 1e-8)

        v_paths[:, t] = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev * dt) * Z2[:, t]
        )
        v_paths[:, t] = np.maximum(v_paths[:, t], 1e-8)

        S_paths[:, t] = (
            S_paths[:, t-1] *
            np.exp((mu - 0.5*v_paths[:, t]) * dt + np.sqrt(v_paths[:, t] * dt) * Z1[:, t])
        )

    preds = S_paths.mean(axis=0)
    return preds, compute_rmse(test_arr, preds), {
        "mu": mu,
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho
    }


# =========================================================
# MODELO 3: MERTON JUMP-DIFFUSION
# =========================================================
def backtest_merton(prices_train, prices_test,
                    lam=2.0, mu_j=-0.02, sigma_j=0.05,
                    n_paths=1000, seed=999):

    train_arr = prices_train.values.astype(float)
    test_arr = prices_test.values.astype(float)

    log_ret = np.log(train_arr[1:] / train_arr[:-1])
    mu = log_ret.mean() * 252
    sigma = log_ret.std() * np.sqrt(252)
    dt = 1 / 252

    S0 = float(train_arr[-1])
    n_steps = len(test_arr)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    Np = rng.poisson(lam * dt, size=(n_paths, n_steps))
    J = rng.normal(mu_j, sigma_j, size=(n_paths, n_steps))

    S_paths = np.zeros((n_paths, n_steps))
    S_paths[:, 0] = S0

    for t in range(1, n_steps):
        S_paths[:, t] = (
            S_paths[:, t-1] *
            np.exp(
                (mu - 0.5*sigma**2)*dt +
                sigma*np.sqrt(dt)*Z[:, t] +
                (Np[:, t] * J[:, t])
            )
        )

    preds = S_paths.mean(axis=0)
    return preds, compute_rmse(test_arr, preds), {
        "mu": mu,
        "sigma": sigma,
        "lambda": lam,
        "mu_j": mu_j,
        "sigma_j": sigma_j
    }


# =========================================================
# INTERFAZ STREAMLIT
# =========================================================
st.set_page_config(page_title="App de Pronóstico DeFi 2", layout="wide")
st.title("App de pronóstico DeFi 2: GBM, Heston y Merton")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker de Yahoo Finance", value="AAPL")
with col2:
    years = st.slider("Años de historia a usar", 1, 10, 5)

end_date = date.today()
start_date = end_date - timedelta(days=365 * years)

st.write(f"Rango de fechas: {start_date} → {end_date}")

test_size = st.slider("Proporción de datos para test", 0.10, 0.50, 0.20)

# Sidebar
st.sidebar.header("Parámetros de modelos (opcional)")

st.sidebar.subheader("Heston")
kappa = st.sidebar.slider("kappa", 0.1, 5.0, 2.0)
theta_factor = st.sidebar.slider("factor θ", 0.5, 2.0, 1.0)
sigma_v = st.sidebar.slider("σ_v", 0.1, 1.5, 0.5)
rho = st.sidebar.slider("ρ", -0.99, 0.0, -0.7)

st.sidebar.subheader("Merton")
lam = st.sidebar.slider("λ saltos/año", 0.1, 5.0, 2.0)
mu_j = st.sidebar.slider("μ_j media del salto", -0.1, 0.1, -0.02)
sigma_j = st.sidebar.slider("σ_j volatilidad del salto", 0.01, 0.3, 0.05)

n_paths = st.sidebar.slider("Número de trayectorias", 200, 3000, 1000, step=100)

# =========================================================
# EJECUCIÓN
# =========================================================
if st.button("Ejecutar pronósticos y backtesting"):

    prices = download_prices(ticker, start_date, end_date)

    if prices is None or len(prices) < 50:
        st.error("No se pudieron descargar datos suficientes.")
    else:
        st.subheader("Serie histórica")
        st.line_chart(prices)

        train, test = train_test_split(prices, test_size=test_size)

        st.write(f"Observaciones: {len(prices)} — Train: {len(train)} — Test: {len(test)}")

        # Ejecutar modelos (RMSE se recalcula afuera)
        gbm_preds, _, gbm_params = backtest_gbm(train, test, n_paths=n_paths)
        heston_preds, _, heston_params = backtest_heston(
            train, test, kappa=kappa, theta_factor=theta_factor,
            sigma_v=sigma_v, rho=rho, n_paths=n_paths
        )
        merton_preds, _, merton_params = backtest_merton(
            train, test, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
            n_paths=n_paths
        )

        # Convertir todo a 1D
        gbm_preds = np.array(gbm_preds).flatten()
        heston_preds = np.array(heston_preds).flatten()
        merton_preds = np.array(merton_preds).flatten()
        test_arr = np.array(test.values).flatten()

        # RMSE corregido
        gbm_rmse = compute_rmse(test_arr, gbm_preds)
        heston_rmse = compute_rmse(test_arr, heston_preds)
        merton_rmse = compute_rmse(test_arr, merton_preds)

        # Tabla RMSE
        rmse_df = pd.DataFrame({
            "Modelo": ["GBM", "Heston", "Merton"],
            "RMSE": [gbm_rmse, heston_rmse, merton_rmse]
        }).set_index("Modelo")

        st.subheader("Resultados de Backtesting (RMSE)")
        st.table(rmse_df.style.format({"RMSE": "{:.4f}"}))

        best_model = rmse_df["RMSE"].idxmin()
        st.success(f"Mejor modelo según RMSE: {best_model}")

        # Gráfico comparación
        st.subheader("Predicciones vs Real")
        comp_df = pd.DataFrame({
            "Real": test_arr,
            "GBM": gbm_preds,
            "Heston": heston_preds,
            "Merton": merton_preds
        }, index=test.index)

        st.line_chart(comp_df)

        # Parámetros
        st.subheader("Parámetros estimados")
        st.write("GBM:", gbm_params)
        st.write("Heston:", heston_params)
        st.write("Merton:", merton_params)
