import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def download_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        return None

    for col in ["Adj Close", "Close", "close", "adjclose"]:
        if col in data.columns:
            return data[col].dropna().astype(float)

    return None


def train_test_split(series, test_size=0.2):
    n = len(series)
    n_test = int(n * test_size)
    return series.iloc[:-n_test], series.iloc[-n_test:]


def compute_rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred)**2))


# =========================================================
# MODELO GBM — CON TODAS LAS TRAYECTORIAS
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
    S_paths[:, 0] = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma*np.sqrt(dt)*Z[:, 0])

    for t in range(1, n_steps):
        S_paths[:, t] = (
            S_paths[:, t-1] *
            np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, t])
        )

    preds = S_paths.mean(axis=0)

    rmse_paths = np.sqrt(np.mean((S_paths - test_arr.reshape(1, -1))**2, axis=1))
    best_idx = rmse_paths.argmin()

    return {
        "paths": S_paths,
        "mean": preds,
        "rmse_mean": compute_rmse(test_arr, preds),
        "rmse_each": rmse_paths,
        "best_traj": S_paths[best_idx],
        "params": {"mu": mu, "sigma": sigma}
    }


# =========================================================
# MODELO HESTON — COMPLETO
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
            v_prev +
            kappa * (theta - v_prev) * dt +
            sigma_v * np.sqrt(v_prev * dt) * Z2[:, t]
        )
        v_paths[:, t] = np.maximum(v_paths[:, t], 1e-8)

        S_paths[:, t] = (
            S_paths[:, t-1] *
            np.exp((mu - 0.5 * v_paths[:, t]) * dt + np.sqrt(v_paths[:, t] * dt) * Z1[:, t])
        )

    preds = S_paths.mean(axis=0)

    rmse_paths = np.sqrt(np.mean((S_paths - test_arr.reshape(1, -1))**2, axis=1))
    best_idx = rmse_paths.argmin()

    return {
        "paths": S_paths,
        "mean": preds,
        "rmse_mean": compute_rmse(test_arr, preds),
        "rmse_each": rmse_paths,
        "best_traj": S_paths[best_idx],
        "params": {
            "mu": mu, "v0": v0, "kappa": kappa,
            "theta": theta, "sigma_v": sigma_v, "rho": rho
        }
    }


# =========================================================
# MODELO MERTON — COMPLETO
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
    Np = rng.poisson(lam * dt, (n_paths, n_steps))
    J = rng.normal(mu_j, sigma_j, (n_paths, n_steps))

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

    rmse_paths = np.sqrt(np.mean((S_paths - test_arr.reshape(1, -1))**2, axis=1))
    best_idx = rmse_paths.argmin()

    return {
        "paths": S_paths,
        "mean": preds,
        "rmse_mean": compute_rmse(test_arr, preds),
        "rmse_each": rmse_paths,
        "best_traj": S_paths[best_idx],
        "params": {
            "mu": mu, "sigma": sigma,
            "lambda": lam, "mu_j": mu_j, "sigma_j": sigma_j
        }
    }


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="App DeFi 2", layout="wide")
st.title("App de pronóstico DeFi 2: GBM, Heston y Merton")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", "AAPL")
with col2:
    years = st.slider("Años de historia", 1, 10, 5)

end_date = date.today()
start_date = end_date - timedelta(days=365 * years)

st.write(f"Rango de fechas: {start_date} → {end_date}")

test_size = st.slider("Proporción test", 0.1, 0.5, 0.2)

# Sidebar
st.sidebar.header("Parámetros")

st.sidebar.subheader("Heston")
kappa = st.sidebar.slider("kappa", 0.1, 5.0, 2.0)
theta_factor = st.sidebar.slider("theta_factor", 0.5, 2.0, 1.0)
sigma_v = st.sidebar.slider("sigma_v", 0.1, 1.5, 0.5)
rho = st.sidebar.slider("rho", -0.99, 0.0, -0.7)

st.sidebar.subheader("Merton")
lam = st.sidebar.slider("lambda", 0.1, 5.0, 2.0)
mu_j = st.sidebar.slider("mu_j", -0.1, 0.1, -0.02)
sigma_j = st.sidebar.slider("sigma_j", 0.01, 0.3, 0.05)

n_paths = st.sidebar.slider("N trayectorias", 200, 3000, 1000, step=100)

modelo_seleccionado = st.selectbox(
    "Modelo para ver trayectorias",
    ["GBM", "Heston", "Merton"]
)

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

        # Ejecutar modelos
        gbm_result = backtest_gbm(train, test, n_paths=n_paths)
        heston_result = backtest_heston(train, test, kappa=kappa,
                                        theta_factor=theta_factor,
                                        sigma_v=sigma_v, rho=rho,
                                        n_paths=n_paths)
        merton_result = backtest_merton(train, test,
                                        lam=lam, mu_j=mu_j,
                                        sigma_j=sigma_j,
                                        n_paths=n_paths)

        # RMSE general
        rmse_df = pd.DataFrame({
            "Modelo": ["GBM", "Heston", "Merton"],
            "RMSE": [
                gbm_result["rmse_mean"],
                heston_result["rmse_mean"],
                merton_result["rmse_mean"]
            ]
        }).set_index("Modelo")

        st.subheader("Resultados de Backtesting (RMSE)")
        st.table(rmse_df.style.format({"RMSE": "{:.4f}"}))

        best_model = rmse_df["RMSE"].idxmin()
        st.success(f"Mejor modelo según RMSE promedio: {best_model}")

        # Elegir modelo para graficar
        if modelo_seleccionado == "GBM":
            res = gbm_result
        elif modelo_seleccionado == "Heston":
            res = heston_result
        else:
            res = merton_result

        # FAN chart (todas las trayectorias)
        paths = res["paths"]
        fan_df = pd.DataFrame(paths.T, index=test.index)

        st.subheader(f"Trayectorias simuladas — {modelo_seleccionado}")
        st.line_chart(fan_df)

        # Comparación de mejor trayectoria
        best_df = pd.DataFrame({
            "Real": test.values,
            "Mejor trayectoria": res["best_traj"],
            "Media": res["mean"]
        }, index=test.index)

        st.subheader(f"Mejor trayectoria vs Real — {modelo_seleccionado}")
        st.line_chart(best_df)

        # Parámetros
        st.subheader("Parámetros estimados")
        st.json(gbm_result["params"])
        st.json(heston_result["params"])
        st.json(merton_result["params"])
