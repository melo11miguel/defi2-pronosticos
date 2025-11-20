import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta
import warnings

# Ignorar advertencias de matplotlib y pandas para una interfaz limpia
warnings.filterwarnings('ignore', category=FutureWarning)

# =========================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Modelos GBM - Heston - Merton",
    layout="wide"
)

st.title("Simulación y Backtesting: GBM, Heston y Merton")
st.markdown(
    "Aplicación para descargar activos de Yahoo Finance, "
    "simular tres modelos estocásticos (GBM, Merton, Heston) y escoger el mejor por **RMSE**."
)

# =========================================================
# FUNCIONES AUXILIARES Y DE DATOS
# =========================================================
@st.cache_data
def download_prices(ticker: str, start: str, end: str) -> pd.Series | None:
    """
    Descarga precios de Yahoo Finance y devuelve la serie de 'Adj Close' (o similar).
    """
    # Descargar data de Yahoo Finance
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    if data.empty:
        return None

    # Intentar obtener la columna de cierre ajustado o cierre
    for col in ["Adj Close", "Close", "close", "adjclose"]:
        if col in data.columns:
            # Drop NaN y asegurar tipo flotante
            s = data[col].dropna()
            if not s.empty:
                return s.astype(float)

    return None


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Calcula los retornos logarítmicos diarios."""
    return np.log(prices / prices.shift(1)).dropna()


def train_test_split_series(series: pd.Series, test_size: float = 0.2):
    """Divide la serie de tiempo en entrenamiento y prueba."""
    n = len(series)
    # Asegurar un mínimo de 5 días para test
    n_test = max(5, int(n * test_size))
    # Asegurar que el set de test no es más grande que el total
    if n_test >= n:
        n_test = max(1, n - 1)

    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]
    return train, test


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la Raíz del Error Cuadrático Medio (RMSE)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# =========================================================
# MODELO 1: MOVIMIENTO GEOMÉTRICO BROWNIANO (GBM)
# =========================================================
def estimate_gbm_params(prices: pd.Series):
    """
    Estima mu y sigma anuales de un GBM a partir de retornos log diarios.
    """
    r = compute_log_returns(prices)
    mu_daily = r.mean()
    sigma_daily = r.std()
    
    # Asunción de 252 días de trading al año
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    return mu_annual, sigma_annual


def simulate_gbm_paths(S0: float, mu: float, sigma: float,
                       dt: float, n_steps: int, n_paths: int, seed: int = 42):
    """Simula trayectorias de precios usando GBM."""
    np.random.seed(seed)
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    # Términos para la discretización de Euler (Log-GBM)
    mu_dt = (mu - 0.5 * sigma ** 2) * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        # S(t) = S(t-1) * exp(drift_term + diffusion_term * Z)
        S[t, :] = S[t - 1, :] * np.exp(mu_dt + sigma_sqrt_dt * z)

    return S


# =========================================================
# MODELO 2: MERTON (JUMP DIFFUSION)
# =========================================================
def estimate_merton_params(prices: pd.Series,
                           jump_threshold_sigma: float = 3.0):
    """
    Estima parámetros de Merton (lambda, mu_J, sigma_J) de forma simple.
    El componente GBM (mu, sigma) se estima igual que el GBM puro.
    """
    r = compute_log_returns(prices)
    mu_daily = r.mean()
    sigma_daily = r.std()

    # 1. Componente GBM
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * np.sqrt(252)

    # 2. Detección de saltos (Saltos > 3 desviaciones estándar)
    z_scores = (r - mu_daily) / sigma_daily
    jumps = r[np.abs(z_scores) > jump_threshold_sigma]

    n_days = len(r)
    years = n_days / 252

    if len(jumps) > 0 and years > 0:
        lam = len(jumps) / years  # Intensidad anual (frecuencia de saltos)
        mu_J = jumps.mean()       # Tamaño medio del salto
        sigma_J = jumps.std() if len(jumps) > 1 else 0.01
    else:
        # Valores de respaldo si no se detectan saltos significativos
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
    """Simula trayectorias de precios usando el modelo de Merton (Jump Diffusion)."""
    np.random.seed(seed)
    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    # Drift ajustado por el componente de saltos (compensación)
    drift = (mu - 0.5 * sigma ** 2 - lam * mu_J) * dt
    diff_coeff = sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        # 1. Difusión (Browniano)
        z = np.random.normal(size=n_paths)
        dW = diff_coeff * z

        # 2. Saltos (Poisson)
        N = np.random.poisson(lam * dt, size=n_paths) # Número de saltos en dt
        J = np.zeros(n_paths)
        
        mask_jumps = N > 0
        if np.any(mask_jumps):
            Nj = N[mask_jumps]
            # La suma de N_j saltos Normales ~ Normal(N_j * mu_J, sqrt(N_j) * sigma_J)
            J[mask_jumps] = np.random.normal(
                loc=Nj * mu_J,
                scale=np.sqrt(Nj) * sigma_J
            )

        log_factor = drift + dW + J
        S[t, :] = S[t - 1, :] * np.exp(log_factor)

    return S


# =========================================================
# MODELO 3: HESTON (VOLATILIDAD ESTOCÁSTICA)
# =========================================================
def estimate_heston_params(prices: pd.Series):
    """
    Estimación simplificada para Heston. Idealmente, los parámetros de Heston 
    (kappa, theta, xi, rho) se estiman usando MLE o métodos GMM, pero aquí 
    usamos valores históricos simples y valores típicos para los parámetros 
    de reversión y vol-of-vol.
    """
    r = compute_log_returns(prices)
    mu_daily = r.mean()
    sigma_daily = r.std()

    mu_annual = mu_daily * 252

    var_daily = sigma_daily ** 2
    v0 = var_daily
    theta = var_daily  # Varianza de largo plazo ~ varianza histórica

    # Parámetros "típicos" (se pueden ajustar)
    kappa = 1.5      # Velocidad de reversión a la media
    xi = 0.5         # Volatilidad de la volatilidad (vol-of-vol)
    rho = -0.7       # Correlación entre el activo y la volatilidad

    return mu_annual, v0, kappa, theta, xi, rho

# Las ecuaciones diferenciales estocásticas de Heston (SDEs) son:
# dS_t = mu S_t dt + sqrt(v_t) S_t dW_{S,t}
# dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_{v,t}
# Donde dW_{S,t} y dW_{v,t} están correlacionados por rho.
# 

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
    """Simula trayectorias de precios usando el modelo de Heston con esquema de Full Truncation."""
    np.random.seed(seed)
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths)) # Trayectorias de la varianza

    S[0, :] = S0
    v[0, :] = v0

    for t in range(1, n_steps + 1):
        # Generar movimientos brownianos independientes
        z1 = np.random.normal(size=n_paths)
        z2 = np.random.normal(size=n_paths)

        # Correlacionar los brownianos
        dW_v = np.sqrt(dt) * z2
        dW_s = np.sqrt(dt) * (rho * z2 + np.sqrt(1 - rho ** 2) * z1)

        # Aplicar Full Truncation para la varianza (evitar sqrt(v) de números negativos)
        v_prev = np.clip(v[t - 1, :], 1e-8, None)

        # 1. Actualización de la Varianza (v)
        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dW_v
        v_t = v_prev + dv
        v_t = np.clip(v_t, 1e-8, None) # Truncamiento completo después del paso

        # 2. Actualización del Precio (S) - usando la SDE del log-precio
        dS = (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW_s
        S[t, :] = S[t - 1, :] * np.exp(dS)
        
        v[t, :] = v_t

    return S, v


# =========================================================
# GRÁFICOS Y VISUALIZACIÓN
# =========================================================
def make_fan_chart(test_index, S_paths, real_prices, title: str):
    """
    Genera una figura de abanico (percentiles) y superpone la serie real.
    S_paths: matriz (n_steps+1, n_paths). Ignoramos t=0 para backtest.
    """
    # El set de prueba tiene n_steps observaciones (t=1 a t=n_steps)
    n_steps = S_paths.shape[0] - 1

    # Calcular percentiles para cada paso de tiempo
    percentiles = np.percentile(S_paths[1:, :], [5, 25, 50, 75, 95], axis=1)
    p5, p25, p50, p75, p95 = percentiles

    fig, ax = plt.subplots(figsize=(10, 5))

    # Abanico (Fan Chart)
    ax.fill_between(test_index, p5, p95, color='#4CAF50', alpha=0.1, label="5%-95%")
    ax.fill_between(test_index, p25, p75, color='#4CAF50', alpha=0.3, label="25%-75%")
    ax.plot(test_index, p50, color='#2196F3', label="Mediana simulada", linewidth=2, linestyle='--')

    # Serie real
    ax.plot(test_index, real_prices.values, color='#E91E63', label="Precio real (Test)", linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Precio", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


# =========================================================
# INTERFAZ DE LA APP (STREAMLIT)
# =========================================================

# --- Sidebar: parámetros de usuario ---
st.sidebar.header("Parámetros de simulación")

default_end = date.today()
default_start = default_end - timedelta(days=5 * 365) # 5 años de datos

ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="AAPL")
start_date = st.sidebar.date_input("Fecha inicio", value=default_start)
end_date = st.sidebar.date_input("Fecha fin", value=default_end)

test_size = st.sidebar.slider("Proporción de datos para test (backtest)",
                              min_value=0.05, max_value=0.5, value=0.2, step=0.05)

n_paths = st.sidebar.slider("Número de trayectorias simuladas",
                            min_value=100, max_value=2000, value=500, step=100)

st.sidebar.markdown(
    """
    ***Notas de la Simulación:***
    * `dt` se asume como $1/252$ (pasos diarios).
    * Los parámetros de Heston y Merton se estiman de forma simplificada a partir de la historia.
    """
)

if st.sidebar.button("Ejecutar modelos", type="primary"):
    if start_date >= end_date:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        st.stop()
    
    # --- Descarga de Datos ---
    with st.spinner(f"Descargando datos para {ticker}..."):
        prices = download_prices(
            ticker=ticker,
            start=str(start_date),
            end=str(end_date)
        )

    if prices is None or len(prices) < 260:
        st.error("No se pudo descargar el activo o hay muy pocos datos (mínimo recomendado: ~260 días).")
        st.stop()

    # --- División de Datos ---
    train_prices, test_prices = train_test_split_series(prices, test_size=test_size)
    
    st.subheader(f"Serie de precios: {ticker}")
    st.line_chart(prices)

    st.markdown("---")
    st.markdown(f"**Parámetros de Backtesting:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Activo Inicial ($S_0$)", f"${train_prices.iloc[-1]:.2f}")
    col2.metric("Período de Entrenamiento", f"{train_prices.index[0].date()} a {train_prices.index[-1].date()} ({len(train_prices)} obs.)")
    col3.metric("Período de Prueba (Backtest)", f"{test_prices.index[0].date()} a {test_prices.index[-1].date()} ({len(test_prices)} obs.)")
    st.markdown("---")


    S0 = train_prices.iloc[-1]
    n_steps = len(test_prices)
    dt = 1 / 252

    results = {}
    
    # -----------------------------------------------------
    # SIMULACIÓN Y BACKTESTING
    # -----------------------------------------------------
    
    # 1. GBM
    with st.expander("Simulación GBM", expanded=True):
        try:
            mu_gbm, sigma_gbm = estimate_gbm_params(train_prices)
            st.code(f"GBM Parámetros (anuales): μ={mu_gbm:.4f}, σ={sigma_gbm:.4f}")
            
            S_gbm = simulate_gbm_paths(
                S0=S0, mu=mu_gbm, sigma=sigma_gbm, dt=dt, n_steps=n_steps, n_paths=n_paths, seed=1
            )
            gbm_mean = S_gbm[1:, :].mean(axis=1) # Usamos la media de las trayectorias como predicción
            rmse_gbm = compute_rmse(test_prices.values, gbm_mean)

            fig_gbm = make_fan_chart(
                test_index=test_prices.index, S_paths=S_gbm, real_prices=test_prices, title=f"GBM Simulación vs. Real"
            )
            st.pyplot(fig_gbm)
            st.info(f"RMSE GBM: {rmse_gbm:.4f}")

            results["GBM"] = {"rmse": rmse_gbm, "fig": fig_gbm}
        except Exception as e:
            st.error(f"Error en el modelo GBM: {e}")

    # 2. MERTON
    with st.expander("Simulación Merton (Jump Diffusion)", expanded=True):
        try:
            mu_mer, sigma_mer, lam_mer, mu_J_mer, sigma_J_mer = estimate_merton_params(train_prices)
            st.code(f"Merton Parámetros (anuales): μ={mu_mer:.4f}, σ_diff={sigma_mer:.4f}, λ={lam_mer:.4f}, μ_J={mu_J_mer:.4f}, σ_J={sigma_J_mer:.4f}")

            S_mer = simulate_merton_paths(
                S0=S0, mu=mu_mer, sigma=sigma_mer, lam=lam_mer, mu_J=mu_J_mer, sigma_J=sigma_J_mer, 
                dt=dt, n_steps=n_steps, n_paths=n_paths, seed=2
            )
            mer_mean = S_mer[1:, :].mean(axis=1)
            rmse_mer = compute_rmse(test_prices.values, mer_mean)

            fig_mer = make_fan_chart(
                test_index=test_prices.index, S_paths=S_mer, real_prices=test_prices, title=f"Merton Simulación vs. Real"
            )
            st.pyplot(fig_mer)
            st.info(f"RMSE Merton: {rmse_mer:.4f}")
            
            results["Merton"] = {"rmse": rmse_mer, "fig": fig_mer}
        except Exception as e:
            st.error(f"Error en el modelo Merton: {e}")

    # 3. HESTON
    with st.expander("Simulación Heston (Volatilidad Estocástica)", expanded=True):
        try:
            mu_h, v0_h, kappa_h, theta_h, xi_h, rho_h = estimate_heston_params(train_prices)
            st.code(f"Heston Parámetros (anuales): μ={mu_h:.4f}, v₀={v0_h:.4f}, κ={kappa_h:.4f}, θ={theta_h:.4f}, ξ={xi_h:.4f}, ρ={rho_h:.4f}")
            
            S_h, v_h = simulate_heston_paths(
                S0=S0, mu=mu_h, v0=v0_h, kappa=kappa_h, theta=theta_h, xi=xi_h, rho=rho_h, 
                dt=dt, n_steps=n_steps, n_paths=n_paths, seed=3
            )
            h_mean = S_h[1:, :].mean(axis=1)
            rmse_h = compute_rmse(test_prices.values, h_mean)

            fig_h = make_fan_chart(
                test_index=test_prices.index, S_paths=S_h, real_prices=test_prices, title=f"Heston Simulación vs. Real"
            )
            st.pyplot(fig_h)
            st.info(f"RMSE Heston: {rmse_h:.4f}")
            
            results["Heston"] = {"rmse": rmse_h, "fig": fig_h}
        except Exception as e:
            st.error(f"Error en el modelo Heston: {e}")

    # =====================================================
    # COMPARACIÓN FINAL DE RESULTADOS
    # =====================================================
    if results:
        st.markdown("## Resumen de Resultados y Selección del Modelo")
        
        # Tabla de RMSE
        rmse_data = {name: info["rmse"] for name, info in results.items()}
        rmse_table = pd.DataFrame(
            {"RMSE": rmse_data}
        ).sort_values("RMSE")
        
        st.subheader("Tabla de RMSE (Root Mean Square Error)")
        st.dataframe(rmse_table.style.format({"RMSE": "{:.4f}"}), use_container_width=True)

        best_model = rmse_table.index[0]
        st.success(f"¡El modelo con el mejor desempeño (menor RMSE) es: **{best_model}**!")

    else:
        st.warning("No se pudieron generar resultados para ningún modelo.")

else:
    st.info("Configura los parámetros del activo y la simulación en el panel lateral (sidebar) y pulsa **'Ejecutar modelos'**.")
