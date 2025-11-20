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
    page_title="Simulación y Backtesting: GBM, Heston, Merton",
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
    n_test = max(5, min(n - 1, n_test))  # al menos 5 observaciones en test
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
    Estima mu y sigma anuales para un GBM a partir de retornos log.
    Se asegura de trabajar siempre con numpy (no con Series).
    """
    r_series = compute_log_returns(prices)
    r = r_series.values.astype(float)

    if r.size < 2:
        return 0.0, 0.2

    mu_daily = r.mean()
    sigma_daily = r.std(ddof=1)

    if sigma_daily == 0 or np.isnan(sigma_daily):
        sigma_daily = 0.01

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
# MODELO MERTON
# =========================================================
def estimate_merton_params(prices: pd.Series,
                           jump_threshold_sigma: float = 3.0):
    """
    Estima parámetros básicos para Merton.
    Todo se hace en numpy para evitar ambigüedad de Series.
    """
    r_series = compute_log_returns(prices)
    r = r_series.values.astype(float)

    if r.size < 2:
        return 0.0, 0.2, 0.1, 0.0, 0.01

    mu_daily = r.mean()
    sigma_daily = r.std(ddof=1)

    if sigma_daily == 0 or np.isnan(sigma_daily):
        sigma_daily = 0.01

    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    # Saltos
    z_scores = (r - mu_daily) / sigma_daily
    jumps = r[np.abs(z_scores) > jump_threshold_sigma]

    n_days = r.size
    years = n_days / 252.0

    if jumps.size > 0 and years > 0:
        lam = jumps.size / years
        mu_J = jumps.mean()
        sigma_J = jumps.std(ddof=1) if jumps.size > 1 else 0.01
    else:
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

    drift = (mu - 0.5 * sigma ** 2 - lam * mu_J) * dt
    diff_coeff = sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        dW = diff_coeff * z

        N = np.random.poisson(lam * dt, size=n_paths)
        J = np.zeros(n_paths)
        mask_jumps = N > 0
        if np.any(mask_jumps):
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
    Estimación simplificada de parámetros Heston.
    Todo en numpy para evitar problemas.
    """
    r_series = compute_log_returns(prices)
    r = r_series.values.astype(float)

    if r.size < 2:
        mu_annual = 0.0
        v0 = 0.04
    else:
        mu_daily = r.mean()
        sigma_daily = r.std(ddof=1)
        if sigma_daily == 0 or np.isnan(sigma_daily):
            sigma_daily = 0.2
        mu_annual = mu_daily * 252
        v0 = sigma_daily ** 2

    theta = v0
    kappa = 1.5
    xi = 0.5
    rho = -0.7
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
    v[0, :] = max(v0, 1e-8)

    for t in range(1, n_steps + 1):
        z1 = np.random.normal(size=n_paths)
        z2 = np.random.normal(size=n_paths)

        dW_v = np.sqrt(dt) * z2
        dW_s = np.sqrt(dt) * (rho * z2 + np.sqrt(1 - rho ** 2) * z1)

        v_prev = np.clip(v[t - 1, :], 1e-8, None)

        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dW_v
        v_t = np.clip(v_prev + dv, 1e-8, None)

        dS = (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW_s
        S[t, :] = S[t - 1, :] * np.exp(dS)
        v[t, :] = v_t

    return S, v


# =========================================================
# GRÁFICO DE ABANICO
# =========================================================
def make_fan_chart(test_index, S_paths, real_prices, title: str):
    n_steps = S_paths.shape[0] - 1
    if n_steps <= 0:
        raise ValueError("Muy pocos pasos para el backtest.")

    percentiles = np.percentile(S_paths[1:, :], [5, 25, 50, 75, 95], axis=1)
    p5, p25, p50, p75, p95 = percentiles

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(test_index, p5, p95, alpha=0.2, label="5%-95%")
    ax.fill_between(test_index, p25, p75, alpha=0.4, label="25%-75%")
    ax.plot(test_index, p50, label="Mediana simulada", linewidth=2)
    ax.plot(test_index, real_prices.values, label="Precio real", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# =========================================================
# SIDEBAR: PARÁMETROS DE USUARIO
# =========================================================
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

# Parámetros manuales
with st.sidebar.expander("Parámetros GBM (opcionales)", expanded=False):
    use_manual_gbm = st.checkbox("Usar parámetros manuales GBM", value=False)
    mu_gbm_manual = st.number_input("mu anual GBM", value=0.10, format="%.5f")
    sigma_gbm_manual = st.number_input("sigma anual GBM", value=0.20,
                                       min_value=0.0001, format="%.5f")

with st.sidebar.expander("Parámetros Merton (opcionales)", expanded=False):
    use_manual_merton = st.checkbox("Usar parámetros manuales Merton", value=False)
    lam_manual = st.number_input("lambda (intensidad saltos)", value=0.10,
                                 min_value=0.0, format="%.5f")
    mu_J_manual = st.number_input("mu_J (tamaño medio salto)", value=0.00, format="%.5f")
    sigma_J_manual = st.number_input("sigma_J (vol salto)", value=0.05,
                                     min_value=0.0001, format="%.5f")

with st.sidebar.expander("Parámetros Heston (opcionales)", expanded=False):
    use_manual_heston = st.checkbox("Usar parámetros manuales Heston", value=False)
    v0_manual = st.number_input("v0 (var inicial)", value=0.04,
                                min_value=0.000001, format="%.6f")
    kappa_manual = st.number_input("kappa (vel. reversión)", value=1.50,
                                   min_value=0.0001, format="%.5f")
    theta_manual = st.number_input("theta (var largo plazo)", value=0.04,
                                   min_value=0.000001, format="%.6f")
    xi_manual = st.number_input("xi (volatilidad de la volatilidad σᵥ)", value=0.50,
                                min_value=0.0001, format="%.5f")
    rho_manual = st.number_input("rho (correlación)", value=-0.70,
                                 min_value=-0.99, max_value=0.99, format="%.3f")

# =========================================================
# EJECUCIÓN
# =========================================================
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
    if len(test_prices) < 5:
        st.error("Muy pocos datos en el conjunto de prueba.")
        st.stop()

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
        mu_gbm_est, sigma_gbm_est = estimate_gbm_params(train_prices)
        mu_gbm = mu_gbm_manual if use_manual_gbm else mu_gbm_est
        sigma_gbm = sigma_gbm_manual if use_manual_gbm else sigma_gbm_est

        st.write(f"GBM - mu estimado: {mu_gbm_est:.4f}, sigma estimado: {sigma_gbm_est:.4f} "
                 f"{'(usando valores manuales)' if use_manual_gbm else ''}")

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
        results["GBM"] = {"rmse": rmse_gbm, "fig": fig_gbm}
    except Exception as e:
        st.error(f"Error en el modelo GBM: {e}")

    # -----------------------------------------------------
    # MERTON
    # -----------------------------------------------------
    try:
        mu_mer_est, sigma_mer_est, lam_est, mu_J_est, sigma_J_est = estimate_merton_params(train_prices)
        lam_mer = lam_manual if use_manual_merton else lam_est
        mu_J_mer = mu_J_manual if use_manual_merton else mu_J_est
        sigma_J_mer = sigma_J_manual if use_manual_merton else sigma_J_est

        st.write(
            f"Merton - mu estimado: {mu_mer_est:.4f}, sigma estimado: {sigma_mer_est:.4f}, "
            f"lambda estimado: {lam_est:.4f}, mu_J estimado: {mu_J_est:.4f}, sigma_J estimado: {sigma_J_est:.4f} "
            f"{'(usando valores manuales)' if use_manual_merton else ''}"
        )

        S_mer = simulate_merton_paths(
            S0=S0,
            mu=mu_mer_est,
            sigma=sigma_mer_est,
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
        results["Merton"] = {"rmse": rmse_mer, "fig": fig_mer}
    except Exception as e:
        st.error(f"Error en el modelo Merton: {e}")

    # -----------------------------------------------------
    # HESTON
    # -----------------------------------------------------
    try:
        mu_h_est, v0_est, kappa_est, theta_est, xi_est, rho_est = estimate_heston_params(train_prices)
        v0_h = v0_manual if use_manual_heston else v0_est
        kappa_h = kappa_manual if use_manual_heston else kappa_est
        theta_h = theta_manual if use_manual_heston else theta_est
        xi_h = xi_manual if use_manual_heston else xi_est
        rho_h = rho_manual if use_manual_heston else rho_est

        st.write(
            f"Heston - mu estimado: {mu_h_est:.4f}, v0 estimado: {v0_est:.6f}, "
            f"kappa estimado: {kappa_est:.4f}, theta estimado: {theta_est:.6f}, "
            f"xi (σᵥ) estimado: {xi_est:.4f}, rho estimado: {rho_est:.3f} "
            f"{'(usando valores manuales)' if use_manual_heston else ''}"
        )

        S_h, v_h = simulate_heston_paths(
            S0=S0,
            mu=mu_h_est,
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
        results["Heston"] = {"rmse": rmse_h, "fig": fig_h}
    except Exception as e:
        st.error(f"Error en el modelo Heston: {e}")

    # =====================================================
    # RESULTADOS
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

    st.subheader("Abanicos de simulación vs. precios reales")
    cols = st.columns(len(results))
    for col, (name, info) in zip(cols, results.items()):
        with col:
            st.markdown(f"#### {name}")
            st.pyplot(info["fig"])

    # =====================================================
    # GUÍA DE PARÁMETROS (EXPLICACIÓN TEÓRICA)
    # =====================================================
    st.subheader("Guía de parámetros de los modelos")

    tab_resumen, tab_gbm, tab_heston, tab_merton = st.tabs(
        ["Resumen visual", "GBM", "Heston", "Merton"]
    )

    with tab_gbm:
        st.markdown(
            """
### 1. Modelo GBM (Geometric Brownian Motion)

**Parámetros fundamentales**

| Parámetro | Significado              | Impacto                                                      |
|----------:|-------------------------|--------------------------------------------------------------|
| μ (mu)    | Retorno promedio diario | Desplaza la tendencia (al alza o a la baja).                |
| σ (sigma) | Volatilidad diaria      | Qué tan “ancho” o variable es el movimiento del precio.     |
| S₀        | Precio inicial          | Punto de partida de las simulaciones.                       |
| n_paths   | Número de simulaciones  | Mientras más grande, más estable la predicción promedio.    |

**¿Qué controla cada uno?**

- **μ** controla la **pendiente** de las trayectorias.  
  Mayor μ → tendencia más marcada hacia arriba (y viceversa).
- **σ** controla la **anchura de la nube** de trayectorias.  
  Mayor σ → simulaciones más dispersas, más riesgo.
- **n_paths** afecta la **suavidad de la media** de predicción.  
  Más trayectorias → la media simulada es más estable.

**En esta app**

- μ y σ se **estiman automáticamente** a partir del histórico.  
- El usuario solo puede:
  - Ajustar **n_paths** (número de trayectorias).
  - Opcionalmente fijar μ y σ manuales en el sidebar.
            """
        )

    with tab_heston:
        st.markdown(
            """
### 2. Modelo de Heston (volatilidad estocástica)

Este modelo introduce una segunda ecuación para la volatilidad, haciendo que la varianza también evolucione en el tiempo.

**Parámetros fundamentales**

| Parámetro | Significado                          | Impacto                                                              |
|----------:|--------------------------------------|----------------------------------------------------------------------|
| κ (kappa) | Velocidad de reversión               | Qué tan rápido la volatilidad vuelve a su promedio.                 |
| θ (theta) | Nivel de largo plazo de volatilidad  | Volatilidad hacia la que tiende el modelo a largo plazo.           |
| σᵥ (xi)   | Volatilidad de la volatilidad        | Cuánto varía la varianza en sí misma (vol-of-vol).                  |
| ρ (rho)   | Correlación precio–volatilidad       | Relación entre movimientos de precio y cambios de volatilidad.      |
| v₀        | Varianza inicial                     | Varianza desde donde arranca el proceso.                            |
| μ         | Retorno promedio diario              | Misma interpretación que en GBM.                                   |
| n_paths   | Número de simulaciones               | Suavidad de la media simulada.                                      |

**Interpretación rápida**

- κ grande → volatilidad **regresiva**: se mueve pero vuelve rápido al promedio.  
- θ grande → mercado estructuralmente **más volátil** (más incertidumbre).  
- σᵥ grande → volatilidad muy **inestable** (“Heston salvaje”), con picos fuertes.  
- ρ negativo (por ejemplo -0.7) → **efecto apalancamiento**:  
  cuando el precio cae, la volatilidad tiende a subir.

En mercados reales de acciones, ρ suele ser **negativo**.

**En esta app**

El usuario puede modificar en el sidebar:

- **kappa_manual (κ)**  
- **theta_manual (θ)**  
- **xi_manual (σᵥ)**  
- **rho_manual (ρ)**  
- (y también v0_manual si lo desea).

Por defecto, estos parámetros se estiman a partir del histórico y se usan valores típicos cuando no hay suficiente información.
            """
        )

    with tab_merton:
        st.markdown(
            """
### 3. Modelo de Merton (Jump–Diffusion)

Añade **saltos repentinos** al precio (crashes o spikes), superpuestos al movimiento continuo tipo GBM.

**Parámetros fundamentales**

| Parámetro | Significado                         | Impacto                                                   |
|----------:|-------------------------------------|-----------------------------------------------------------|
| λ (lambda)| Frecuencia de saltos por año        | Cuántos saltos ocurren por año en promedio.              |
| μⱼ (mu_j) | Tamaño medio del salto              | Si es negativo → saltos bajistas (crashes).              |
| σⱼ        | Volatilidad del salto               | Variabilidad en el tamaño de los saltos.                 |
| μ         | Retorno “normal” (componente difusa)| Tendencia general del activo.                            |
| σ         | Volatilidad “normal”                | Ruido continuo tipo GBM.                                 |
| n_paths   | Número de trayectorias              | Suavidad de la media de predicción.                      |

**Interpretación**

- λ grande → muchos saltos por año.  
- μⱼ < 0 → saltos típicamente **bajistas**.  
- σⱼ grande → saltos muy **impredecibles** (cola muy pesada).

Este modelo es adecuado cuando:

- Hay caídas rápidas.  
- Hay noticias fuertes o eventos discretos.  
- Se observan **gaps** importantes entre un día y el siguiente.

**En esta app**

El usuario puede ajustar en el sidebar:

- **lam_manual (λ)**  
- **mu_J_manual (μⱼ)**  
- **sigma_J_manual (σⱼ)**  

μ y σ del componente continuo se estiman automáticamente del histórico, igual que en GBM.
            """
        )

    with tab_resumen:
        st.markdown(
            """
### Resumen visual – ¿qué parámetros importan por modelo?

| Modelo  | Parámetros internos principales                     | Parámetros que puede manipular el usuario en esta app |
|---------|------------------------------------------------------|-------------------------------------------------------|
| **GBM** | μ, σ, S₀                                             | n_paths, (opcional) μ y σ                             |
| **Heston** | μ, v₀, θ, κ, σᵥ (xi), ρ                          | κ, θ, σᵥ (xi), ρ, v₀                                  |
| **Merton** | μ, σ, λ, μⱼ, σⱼ                                  | λ, μⱼ, σⱼ                                           |

La idea es que la app sirva tanto para **simular y comparar modelos** como para **entender el papel de cada parámetro** en la forma del abanico y en el RMSE.
            """
        )

else:
    st.info("Configura los parámetros en el sidebar y pulsa **'Ejecutar modelos'**.")
