# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

st.set_page_config(layout="wide", page_title="Modelos: GBM / Heston / Merton")

# -------------------------
# Helpers / matemáticas
# -------------------------
def download_data(ticker, start, end, interval="1d"):
    df = yf.download(ticker, start=start, end=end, progress=False, interval=interval)
    df = df[['Adj Close']].rename(columns={'Adj Close': 'close'}).dropna()
    return df

def returns_log(prices):
    return np.log(prices).diff().dropna()

# GBM sim (vectorizado)
def simulate_gbm(S0, mu, sigma, n_steps, n_paths, dt=1/252, random_state=None):
    rng = np.random.default_rng(random_state)
    normals = rng.standard_normal(size=(n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * normals
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_paths)
    S = np.hstack([np.full((n_paths,1), S0), S])
    return S

# Merton jump-diffusion simulation
def simulate_merton(S0, mu, sigma, lam, mu_j, sigma_j, n_steps, n_paths, dt=1/252, random_state=None):
    rng = np.random.default_rng(random_state)
    S = np.zeros((n_paths, n_steps+1))
    S[:,0] = S0
    for t in range(1, n_steps+1):
        Z = rng.standard_normal(n_paths)
        # Poisson jumps
        Nj = rng.poisson(lam*dt, size=n_paths)
        # jump multiplier
        J = np.exp(mu_j + sigma_j * rng.standard_normal(n_paths))**Nj  # if Nj=0 -> 1
        S[:,t] = S[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z) * J
    return S

# Heston simulation (Euler discretization)
def simulate_heston(S0, mu, v0, kappa, theta, xi, rho, n_steps, n_paths, dt=1/252, random_state=None):
    rng = np.random.default_rng(random_state)
    S = np.zeros((n_paths, n_steps+1))
    v = np.zeros((n_paths, n_steps+1))
    S[:,0] = S0
    v[:,0] = v0
    for t in range(1, n_steps+1):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W1 = Z1
        # correlate: Z2_corr = rho * Z1 + sqrt(1-rho^2) * Z2
        Z2_corr = rho * Z1 + np.sqrt(max(0,1-rho**2)) * Z2
        # variance step (Euler, ensure non-negativity)
        v_prev = v[:,t-1]
        dv = kappa*(theta - v_prev)*dt + xi*np.sqrt(np.maximum(v_prev,0))*np.sqrt(dt)*Z2_corr
        v_new = np.maximum(v_prev + dv, 1e-8)
        v[:,t] = v_new
        # price step
        dS = (mu - 0.5*v_prev)*dt + np.sqrt(np.maximum(v_prev,0))*np.sqrt(dt)*Z1
        S[:,t] = S[:,t-1] * np.exp(dS)
    return S

# RMSE between mean simulated path and actual prices (on same grid)
def rmse_sim_vs_actual(sim_paths, actual_prices):
    mean_sim = np.mean(sim_paths, axis=0)
    # ensure same length
    actual = np.asarray(actual_prices)
    if len(mean_sim) != len(actual):
        raise ValueError("Lengths differ")
    return np.sqrt(np.mean((mean_sim - actual)**2))

# -------------------------
# Calibration heuristics + optimization
# -------------------------
def estimate_gbm_params(prices, trading_days_per_year=252):
    r = returns_log(prices)
    mu_daily = r.mean()
    sigma_daily = r.std(ddof=1)
    mu = mu_daily * trading_days_per_year
    sigma = sigma_daily * np.sqrt(trading_days_per_year)
    return float(mu), float(sigma)

# Objective wrappers for optimization: minimize RMSE for given model parameters
def objective_heston(params, S0, mu, prices_array, n_paths, dt):
    kappa, theta, xi, rho = params
    # fixed v0 = sample var of daily returns * trading_days
    v0 = np.var(returns_log(pd.Series(prices_array))) * 252
    n_steps = len(prices_array)-1
    sim = simulate_heston(S0, mu, v0, kappa, theta, xi, rho, n_steps, n_paths, dt=dt, random_state=42)
    # sim includes S0 at index 0 and subsequent
    sim_mean_with_S0 = np.mean(sim, axis=0)
    return float(rmse_sim_vs_actual(sim, prices_array))

def objective_merton(params, S0, mu, prices_array, n_paths, dt):
    lam, mu_j, sigma_j, sigma = params
    n_steps = len(prices_array)-1
    sim = simulate_merton(S0, mu, sigma, lam, mu_j, sigma_j, n_steps, n_paths, dt=dt, random_state=42)
    return float(rmse_sim_vs_actual(sim, prices_array))

# -------------------------
# Streamlit UI
# -------------------------
st.title("Modelos estocásticos: GBM · Heston · Merton — simulación y backtest")
st.markdown("Sigue: selecciona ticker, periodo, y corre la calibración + simulación. La app devolverá los RMSE y recomendará el mejor modelo.")

col1, col2 = st.columns([1,2])

with col1:
    ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL")
    end_date = st.date_input("Fecha final", value=datetime.today().date())
    start_date = st.date_input("Fecha inicial", value=(datetime.today() - timedelta(days=365*2)).date())
    n_paths = st.number_input("Número de paths por modelo (simulaciones)", min_value=50, max_value=5000, value=500, step=50)
    trading_days = st.number_input("Días de trading por año", min_value=250, max_value=260, value=252)
    n_steps_days = st.number_input("Número de días a simular (horizon, pasos)", min_value=30, max_value=252*2, value=252)
    run_btn = st.button("Correr calibración y simulación")

with col2:
    st.markdown("**Instrucciones rápidas**:")
    st.markdown("- La calibración inicial usa estimadores sencillos. Para Heston y Merton se permite optimizar parámetros mínimos que minimicen RMSE entre la media simulada y precios reales.")
    st.markdown("- El RMSE es calculado entre la *media* de muchas simulaciones y la serie de precios reales en el mismo grid.")
    st.markdown("- Puedes ajustar número de simulaciones para estabilidad.")

if run_btn:
    with st.spinner("Descargando datos y corriendo simulaciones..."):
        df = download_data(ticker, start_date, end_date)
        if df.empty:
            st.error("No se encontraron datos para ese ticker/periodo.")
        else:
            prices = df['close']
            prices = prices[-(n_steps_days+1):]  # usar última ventana con longitud = n_steps+1
            if len(prices) < 60:
                st.warning("Serie muy corta — usar al menos 60 días para calibración más estable.")
            S0 = float(prices.iloc[0])
            dt = 1/trading_days

            # --- GBM params & sim
            mu_gbm, sigma_gbm = estimate_gbm_params(prices, trading_days_per_year=trading_days)
            sim_gbm = simulate_gbm(S0, mu_gbm, sigma_gbm, n_steps_days, n_paths, dt=dt, random_state=123)
            rmse_gbm = rmse_sim_vs_actual(sim_gbm, prices.values)

            # --- Merton: heuristic initial params + optimization
            # heuristic: lam = freq of extreme returns, mu_j,sigma_j from extremes
            r = returns_log(prices)
            daily_sigma = r.std(ddof=1)
            extreme_mask = np.abs(r) > 2.5 * daily_sigma
            lam0 = max(1e-4, extreme_mask.sum() / len(r) * trading_days)  # annualized freq approx
            if extreme_mask.sum() >= 2:
                mu_j0 = r[extreme_mask].mean()
                sigma_j0 = r[extreme_mask].std(ddof=1)
            else:
                mu_j0 = -0.02
                sigma_j0 = 0.05
            sigma0 = daily_sigma * np.sqrt(trading_days)
            # bounds and optimize
            bounds_m = [(1e-6, 10.0), (-1.0, 1.0), (1e-4, 1.0), (1e-6, 2.0)]
            x0 = [lam0, mu_j0, sigma_j0, sigma0]
            try:
                res_m = minimize(objective_merton, x0, args=(S0, mu_gbm, prices.values, n_paths, dt),
                                 bounds=bounds_m, method='L-BFGS-B', options={'maxiter':50})
                lam_opt, mu_j_opt, sigma_j_opt, sigma_m_opt = res_m.x
            except Exception as e:
                lam_opt, mu_j_opt, sigma_j_opt, sigma_m_opt = x0
            sim_merton = simulate_merton(S0, mu_gbm, sigma_m_opt, lam_opt, mu_j_opt, sigma_j_opt, n_steps_days, n_paths, dt=dt, random_state=456)
            rmse_merton = rmse_sim_vs_actual(sim_merton, prices.values)

            # --- Heston: heuristic initial + optimize
            # heuristic initial:
            v0 = np.var(returns_log(prices)) * trading_days
            kappa0 = 1.0
            theta0 = v0
            xi0 = 0.3
            rho0 = -0.5
            x0h = [kappa0, theta0, xi0, rho0]
            bounds_h = [(1e-4, 10.0), (1e-8, 2.0), (1e-4, 2.0), (-0.99, 0.99)]
            try:
                res_h = minimize(lambda x: objective_heston(x, S0, mu_gbm, prices.values, n_paths, dt),
                                 x0h, bounds=bounds_h, method='L-BFGS-B', options={'maxiter':40})
                kappa_opt, theta_opt, xi_opt, rho_opt = res_h.x
            except Exception as e:
                kappa_opt, theta_opt, xi_opt, rho_opt = x0h
            sim_heston = simulate_heston(S0, mu_gbm, v0, kappa_opt, theta_opt, xi_opt, rho_opt, n_steps_days, n_paths, dt=dt, random_state=789)
            rmse_heston = rmse_sim_vs_actual(sim_heston, prices.values)

            # Results table
            results = pd.DataFrame({
                "modelo": ["GBM", "Merton", "Heston"],
                "rmse": [rmse_gbm, rmse_merton, rmse_heston]
            }).sort_values("rmse")
            st.subheader("Resultados — RMSE (menor mejor)")
            st.table(results.style.format({"rmse":"{:.4f}"}))

            best_model = results.iloc[0]['modelo']
            st.success(f"Mejor modelo: {best_model} (RMSE = {results.iloc[0]['rmse']:.4f})")

            # Mostrar parámetros
            st.subheader("Parámetros estimados / optimizados")
            st.write("GBM: mu (anual) = {:.4f}, sigma (anual) = {:.4f}".format(mu_gbm, sigma_gbm))
            st.write("Merton (opt): lambda = {:.4f}, mu_j = {:.4f}, sigma_j = {:.4f}, sigma(diffusion) = {:.4f}".format(
                lam_opt, mu_j_opt, sigma_j_opt, sigma_m_opt))
            st.write("Heston (opt): kappa = {:.4f}, theta = {:.6f}, xi = {:.4f}, rho = {:.4f}, v0 = {:.6f}".format(
                kappa_opt, theta_opt, xi_opt, rho_opt, v0))

            # Plots: precios reales + medias simuladas
            fig, ax = plt.subplots(1,1, figsize=(10,5))
            days = np.arange(len(prices))
            ax.plot(days, prices.values, label="Precio real", linewidth=2)
            ax.plot(days, np.mean(sim_gbm, axis=0), label="GBM (mean sim)")
            ax.plot(days, np.mean(sim_merton, axis=0), label="Merton (mean sim)")
            ax.plot(days, np.mean(sim_heston, axis=0), label="Heston (mean sim)")
            ax.set_xlabel("Días (index)")
            ax.set_ylabel("Precio")
            ax.legend()
            st.pyplot(fig)

            # Mostrar algunos paths sample
            st.subheader("Ejemplo de paths simulados (5 paths aleatorios por modelo)")
            idxs = np.random.choice(n_paths, size=min(5, n_paths), replace=False)
            fig2, ax2 = plt.subplots(1,3, figsize=(15,4))
            for i in idxs:
                ax2[0].plot(sim_gbm[i,:], alpha=0.7)
            ax2[0].set_title("GBM samples")
            for i in idxs:
                ax2[1].plot(sim_merton[i,:], alpha=0.7)
            ax2[1].set_title("Merton samples")
            for i in idxs:
                ax2[2].plot(sim_heston[i,:], alpha=0.7)
            ax2[2].set_title("Heston samples")
            st.pyplot(fig2)

            # Export results
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar resultados (CSV)", csv, file_name=f"results_{ticker}.csv", mime="text/csv")
            st.success("Listo — revisa parámetros y gráficos. Para mejorar la calibración, aumenta n_paths o permite más iteraciones al optimizador.")
