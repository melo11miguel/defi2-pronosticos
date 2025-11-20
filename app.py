import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Modelos GBM - Heston - Merton",
    layout="wide"
)

st.title("Simulación y Backtesting: GBM, Heston y Merton")
st.markdown(
    "Aplicación para descargar activos de Yahoo Finance, "
    "simular tres modelos estocásticos y escoger el mejor por **RMSE**."
)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
@st.cache_data
def download_prices(ticker: str, start: str, end: str) -> pd.Series | None:
    """
    Descarga precios de Yahoo Finance y devuelve la serie de 'Adj Close' (o similar).
    """
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        return None

    for col in ["Adj Close", "Close", "close", "adjclose"]:
        if col in data.columns:
            s = data[col].dropna()
            if not s.empty:
                return s.astype(float)

    return None


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def train_test_split_series(series: pd.Series, test_size: float = 0.2):
    n = len(series)
    n_test = int(n * test_size)
    if n_test < 5:
        n_test = 5
    if n_test >= n:
        n_test = max(1, n - 1)

    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]
    return train, test


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# =========================================================
# MODELO GBM
# =========================================================
def estimate_gbm_params(prices: pd.Series):
    """
    Estima mu y sigma anuales de un GBM a partir de retornos log diarios.
    """
    r = compute_log_returns(prices)
    mu_daily = r.mean()
    sigma_daily = r.std()

    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    return mu_annual, sigma_annual


def simulate_gbm_paths(S0: float, mu: float, sigma: float,
                       dt: float, n_steps: int, n_paths: int, seed: int = 42):
    np.random.seed(seed)
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    mu_dt = (mu - 0.5 * sigma ** 2) * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        S[t, :] = S[t - 1, :] * np.exp(mu_dt + sigma_sqrt_dt * z)

    return S


# =========================================================
# MODELO MERTON (JUMP DIFFUSION)
# =========================================================
def estimate_merton_params(prices: pd.Series,
                           jump_threshold_sigma: float = 3.0):
    """
    Estimación muy simple para Merton:
    - mu y sigma del componente continuo (como GBM)
    - lambda, mu_J, sigma_J para saltos usando un umbral de sigma.
    """
    r = compute_log_returns(prices)
    mu_daily = r.mean()
    sigma_daily = r.std()

    # Estimación de GBM
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    # Detección de saltos
    z_scores = (r - mu_daily) / sigma_daily
    jumps = r[np.abs(z_scores) > jump_threshold_sigma]

    n_days = len(r)
    years = n_days / 252

    if len(jumps) > 0 and years > 0:
        lam = len(jumps) / years  # intensidad anual
        mu_J = jumps.mean()       # tamaño medio del salto (en retornos diarios)
        sigma_J = jumps.std() if len(jumps) > 1 else 0.01
    else:
        # Si no detecta saltos, ponemos valores pequeños
        lam = 0.1
        mu_J = 0.0
        sigma_J = 0.01

    return mu_annual, sigma_annual, lam, mu_J, sigma_J


def simulate_merton_paths(S0: float,
                          mu: float,
                          sigma: float,
                          lam: float,
                          mu_J: float,
                          sigma_J: float,
                          dt: float,
                          n_steps: int,
                          n_paths: int,
                          seed: int = 123):
    np.random.seed(seed)
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    # Drif corregido por componente de saltos
    drift = (mu - 0.5 * sigma ** 2 - lam * mu_J) * dt
    diff_coeff = sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        # Difusión
        z = np.random.normal(size=n_paths)
        dW = diff_coeff * z

        # Saltos (Poisson)
        N = np.random.poisson(lam * dt, size=n_paths)
        J = np.zeros(n_paths)
        # Si hay saltos, sumamos Normal(mu_J, sigma_J) N veces
        mask_jumps = N > 0
        if np.any(mask_jumps):
            # Para cada camino con N>0, suma de N Normales ~ Normal(N*mu_J, sqrt(N)*sigma_J)
            Nj = N[mask_jumps]
            J[mask_jumps] = np.random.normal(
                loc=Nj * mu_J,
                scale=np.sqrt(Nj) * sigma_J
            )

        log_factor = drift + dW + J
        S[t, :] = S[t - 1, :] * np.exp(log_factor)

    return S


# =========================================================
# MODELO HESTON
# =========================================================
def estimate_heston_params(prices: pd.Series):
    """
    Estimación muy simplificada para Heston:
    - mu anual: como GBM
    - v0, theta: varianza media
    - kappa, xi, rho: parámetros fijos razonables
    Esta parte se puede sofisticar más si lo necesitas.
    """
    r = compute_log_returns(prices)
    mu_daily = r.mean()
    sigma_daily = r.std()

    mu_annual = mu_daily * 252

    var_daily = sigma_daily ** 2
    v0 = var_daily
    theta = var_daily  # varianza de largo plazo ~ varianza histórica

    # Parámetros "típicos"
    kappa = 1.5   # velocidad de reversión
    xi = 0.5      # vol-of-vol
    rho = -0.7    # correlación negativa

    return mu_annual, v0, kappa, theta, xi, rho


def simulate_heston_paths(S0: float,
                          mu: float,
                          v0: float,
                          kappa: float,
                          theta: float,
                          xi: float,
                          rho: float,
                          dt: float,
                          n_steps: int,
                          n_paths: int,
                          seed: int = 999):
    np.random.seed(seed)
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0, :] = S0
    v[0, :] = v0

    for t in range(1, n_steps + 1):
        z1 = np.random.normal(size=n_paths)
        z2 = np.random.normal(size=n_paths)

        # Brownianos correlacionados
        dW_v = np.sqrt(dt) * z2
        dW_s = np.sqrt(dt) * (rho * z2 + np.sqrt(1 - rho ** 2) * z1)

        # Full truncation para varianza
        v_prev = np.clip(v[t - 1, :], 1e-8, None)

        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dW_v
        v_t = v_prev + dv
        v_t = np.clip(v_t, 1e-8, None)

        dS = (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW_s
        S[t, :] = S[t - 1, :] * np.exp(dS)
        v[t, :] = v_t

    return S, v


# =========================================================
# GRÁFICO DE ABANICO
# =========================================================
def make_fan_chart(test_index, S_paths, real_prices, title: str):
    """
    Genera una figura de abanico (percentiles) y serie real.
    S_paths: matriz (n_steps+1, n_paths) -> ignoramos t=0 para backtest.
    """
    n_steps = S_paths.shape[0] - 1

    # Percentiles al nivel de cada tiempo
    percentiles = np.percentile(S_paths[1:, :], [5, 25, 50, 75, 95], axis=1)
    p5, p25, p50, p75, p95 = percentiles

    fig, ax = plt.subplots(figsize=(10, 5))

    # Abanico
    ax.fill_between(test_index, p5, p95, alpha=0.2, label="5%-95%")
    ax.fill_between(test_index, p25, p75, alpha=0.4, label="25%-75%")
    ax.plot(test_index, p50, label="Mediana simulada", linewidth=2)

    # Serie real
    ax.plot(test_index, real_prices.values, label="Precio real", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# =========================================================
# INTERFAZ DE LA APP
# =========================================================
# Sidebar: parámetros de usuario
st.sidebar.header("Parámetros de simulación")

default_end = date.today()
default_start = default_end - timedelta(days=5 * 365)

ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="AAPL")
start_date = st.sidebar.date_input("Fecha inicio", value=default_start)
end_date = st.sidebar.date_input("Fecha fin", value=default_end)

test_size = st.sidebar.slider("Proporción de datos para test (backtest)",
                              min_value=0.1, max_value=0.4, value=0.2, step=0.05)

n_paths = st.sidebar.slider("Número de trayectorias simuladas",
                            min_value=200, max_value=2000, value=500, step=100)

st.sidebar.markdown("dt = 1/252 (supuesto de datos diarios)")

if st.sidebar.button("Ejecutar modelos"):
    if start_date >= end_date:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        st.stop()

    prices = download_prices(
        ticker=ticker,
        start=str(start_date),
        end=str(end_date)
    )

    if prices is None or len(prices) < 260:
        st.error("No se pudo descargar el activo o hay muy pocos datos (mínimo ~260 días).")
        st.stop()

    st.subheader(f"Serie de precios: {ticker}")
    st.line_chart(prices)

    train_prices, test_prices = train_test_split_series(prices, test_size=test_size)
    st.write(f"Datos de entrenamiento: {train_prices.index[0].date()} – {train_prices.index[-1].date()}")
    st.write(f"Datos de prueba (backtest): {test_prices.index[0].date()} – {test_prices.index[-1].date()}")
    st.write(f"Número de observaciones (train/test): {len(train_prices)} / {len(test_prices)}")

    S0 = train_prices.iloc[-1]
    n_steps = len(test_prices)
    dt = 1 / 252

    results = {}

    # -----------------------------------------------------
    # GBM
    # -----------------------------------------------------
    try:
        mu_gbm, sigma_gbm = estimate_gbm_params(train_prices)
        S_gbm = simulate_gbm_paths(
            S0=S0,
            mu=mu_gbm,
            sigma=sigma_gbm,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=1
        )
        gbm_mean = S_gbm[1:, :].mean(axis=1)
        rmse_gbm = compute_rmse(test_prices.values, gbm_mean)

        fig_gbm = make_fan_chart(
            test_index=test_prices.index,
            S_paths=S_gbm,
            real_prices=test_prices,
            title=f"Abanico GBM - {ticker}"
        )

        results["GBM"] = {
            "rmse": rmse_gbm,
            "fig": fig_gbm
        }
    except Exception as e:
        st.error(f"Error en el modelo GBM: {e}")

    # -----------------------------------------------------
    # MERTON
    # -----------------------------------------------------
    try:
        mu_mer, sigma_mer, lam_mer, mu_J_mer, sigma_J_mer = estimate_merton_params(train_prices)
        S_mer = simulate_merton_paths(
            S0=S0,
            mu=mu_mer,
            sigma=sigma_mer,
            lam=lam_mer,
            mu_J=mu_J_mer,
            sigma_J=sigma_J_mer,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=2
        )
        mer_mean = S_mer[1:, :].mean(axis=1)
        rmse_mer = compute_rmse(test_prices.values, mer_mean)

        fig_mer = make_fan_chart(
            test_index=test_prices.index,
            S_paths=S_mer,
            real_prices=test_prices,
            title=f"Abanico Merton - {ticker}"
        )

        results["Merton"] = {
            "rmse": rmse_mer,
            "fig": fig_mer
        }
    except Exception as e:
        st.error(f"Error en el modelo Merton: {e}")

    # -----------------------------------------------------
    # HESTON
    # -----------------------------------------------------
    try:
        mu_h, v0_h, kappa_h, theta_h, xi_h, rho_h = estimate_heston_params(train_prices)
        S_h, v_h = simulate_heston_paths(
            S0=S0,
            mu=mu_h,
            v0=v0_h,
            kappa=kappa_h,
            theta=theta_h,
            xi=xi_h,
            rho=rho_h,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=3
        )
        h_mean = S_h[1:, :].mean(axis=1)
        rmse_h = compute_rmse(test_prices.values, h_mean)

        fig_h = make_fan_chart(
            test_index=test_prices.index,
            S_paths=S_h,
            real_prices=test_prices,
            title=f"Abanico Heston - {ticker}"
        )

        results["Heston"] = {
            "rmse": rmse_h,
            "fig": fig_h
        }
    except Exception as e:
        st.error(f"Error en el modelo Heston: {e}")

    # =====================================================
    # MOSTRAR RESULTADOS
    # =====================================================
    if not results:
        st.error("Ningún modelo pudo ejecutarse.")
        st.stop()

    st.subheader("RMSE de cada modelo (backtest)")
    rmse_table = pd.DataFrame(
        {name: {"RMSE": info["rmse"]} for name, info in results.items()}
    ).T.sort_values("RMSE")
    st.table(rmse_table.style.format({"RMSE": "{:.4f}"}))

    best_model = rmse_table.index[0]
    st.success(f"Mejor modelo según RMSE: **{best_model}**")

    # Gráficos de abanico
    st.subheader("Abanicos de simulación vs. precios reales")

    cols = st.columns(len(results))
    for col, (name, info) in zip(cols, results.items()):
        with col:
            st.markdown(f"#### {name}")
            st.pyplot(info["fig"])

else:
    st.info("Configura los parámetros en el sidebar y pulsa **'Ejecutar modelos'**.")


