import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import minimize
from math import sqrt
import warnings

# Ignorar advertencias de optimización para un output más limpio
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. FUNCIONES DE DESCARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

def fetch_data(ticker, start_date, end_date):
    """
    Descarga los datos históricos del precio de cierre de un activo.

    :param ticker: Símbolo del activo (ej: 'AAPL', '^GSPC').
    :param start_date: Fecha de inicio (YYYY-MM-DD).
    :param end_date: Fecha de fin (YYYY-MM-DD).
    :return: DataFrame de pandas con el precio de cierre.
    """
    print(f"Descargando datos para {ticker} desde {start_date} hasta {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No se encontraron datos para el ticker o el rango de fechas especificado.")
        return data['Adj Close']
    except Exception as e:
        print(f"Error al descargar datos: {e}")
        return None

def calculate_log_returns(prices):
    """
    Calcula los retornos logarítmicos diarios.
    """
    return np.log(prices / prices.shift(1)).dropna()

# ==============================================================================
# 2. CALIBRACIÓN DE PARÁMETROS BÁSICOS
# ==============================================================================

def get_basic_parameters(log_returns, dt=1/252):
    """
    Calcula la deriva (mu) y la volatilidad (sigma) anualizadas.
    
    :param dt: Intervalo de tiempo, 1/252 para días hábiles.
    :return: mu, sigma
    """
    # Media de los retornos logarítmicos
    mu_daily = log_returns.mean()
    # Desviación estándar de los retornos logarítmicos
    sigma_daily = log_returns.std()
    
    # Anualización
    mu = mu_daily / dt + 0.5 * sigma_daily**2 / dt # Deriva (drift) ajustada a la media del activo
    sigma = sigma_daily / np.sqrt(dt) # Volatilidad anual
    
    # Parámetros para la simulación
    r = mu_daily / dt # Usamos el drift del activo como tasa de riesgo neutral efectiva
    v0 = sigma_daily # Volatilidad diaria
    
    print(f"\n--- Parámetros Calibrados (Anualizados) ---")
    print(f"Deriva (mu): {mu:.4f}")
    print(f"Volatilidad (sigma): {sigma:.4f}")
    print(f"-------------------------------------------")

    # Devolvemos los parámetros calibrados anualmente (mu, sigma) y los
    # parámetros diarios para la simulación (r_daily, v_daily)
    return mu, sigma, mu_daily, sigma_daily

# ==============================================================================
# 3. SIMULACIÓN DE MODELOS ESTOCÁSTICOS (RISK-NEUTRAL)
# ==============================================================================

def simulate_gbm(S0, mu, sigma, T, N, M=1):
    """
    Simulación del Movimiento Browniano Geométrico (MBG).
    
    dS_t = mu * S_t * dt + sigma * S_t * dW_t
    
    :param S0: Precio inicial.
    :param mu: Deriva anual (drift).
    :param sigma: Volatilidad anual.
    :param T: Tiempo total de simulación (en años).
    :param N: Número de pasos de tiempo.
    :param M: Número de trayectorias (simulaciones).
    :return: Matriz de precios simulados (N+1 x M).
    """
    dt = T / N
    # Usamos la fórmula de Euler-Maruyama discreta (con el término de corrección)
    dW = norm.rvs(size=(N, M), scale=np.sqrt(dt)) 
    
    S = np.zeros((N + 1, M))
    S[0] = S0
    
    # La tasa de riesgo-neutral (mu - 0.5 * sigma^2) es más adecuada para simular,
    # pero usamos el mu directo si los parámetros ya están ajustados.
    
    # Ecuación de simulación: S[i] = S[i-1] * exp((mu - 0.5 * sigma^2) * dt + sigma * dW_i)
    # Por simplicidad en el backtesting, usamos la formulación directa de Euler:
    for t in range(1, N + 1):
        # Fórmula discreta del GBM (Euler-Maruyama)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
        
    return S

# ------------------------------------------------------------------------------
# CALIBRACIÓN Y SIMULACIÓN DE MERTON (JUMP-DIFFUSION)
# ------------------------------------------------------------------------------

def calibration_merton(params, log_returns):
    """Función objetivo para calibrar el modelo de Merton (minimizar la diferencia
    entre la distribución empírica de retornos y la distribución de Merton).
    Aquí usamos la varianza como proxy simple para el ejemplo."""
    
    # Parámetros a estimar: [lambda (frecuencia), mu_J (media del salto), sigma_J (desviación del salto)]
    # Asumimos que mu y sigma_difusión vienen del MBG.
    
    lambda_p, mu_J, sigma_J = params
    
    # Varianza teórica de Merton: Var(r) = sigma_difusión^2 + lambda_p * (mu_J^2 + sigma_J^2)
    # Calibramos solo la parte de salto, asumiendo la varianza de difusión como constante.
    
    # Para la simpleza de este ejemplo, devolveremos un valor grande para evitar optimización compleja
    # La calibración real de Merton requiere MLE (Máxima Verosimilitud) compleja.
    # En un entorno real, no se haría de esta forma simplificada.
    
    return np.mean((log_returns.var() - (0.01 + lambda_p * (mu_J**2 + sigma_J**2)))**2) * 1000 

def calibrate_merton_simple(log_returns):
    """Calibración simple de Merton (valores iniciales razonables)."""
    # Valores iniciales: lambda (1 salto al año), mu_J (media del 0%), sigma_J (desviación del 1%)
    initial_params = [1.0, 0.0, 0.01]
    
    # Solo devolveremos los valores iniciales para evitar una optimización inestable
    # La calibración real de Merton es muy intensiva.
    # res = minimize(calibration_merton, initial_params, args=(log_returns,), method='L-BFGS-B')
    # lambda_p, mu_J, sigma_J = res.x
    
    lambda_p, mu_J, sigma_J = initial_params
    
    print(f"\n--- Parámetros Calibrados de Merton (Iniciales Simples) ---")
    print(f"Frecuencia de salto (lambda): {lambda_p:.4f}")
    print(f"Media del salto (mu_J): {mu_J:.4f}")
    print(f"Desviación estándar del salto (sigma_J): {sigma_J:.4f}")
    print(f"----------------------------------------------------------")
    return lambda_p, mu_J, sigma_J


def simulate_merton(S0, mu, sigma, lambda_p, mu_J, sigma_J, T, N, M=1):
    """
    Simulación del Modelo de Salto-Difusión de Merton.
    
    dS_t = S_t * (mu - lambda * E[J]) * dt + S_t * sigma * dW_t + S_{t-} * dJ_t
    
    :param lambda_p: Frecuencia de saltos por año (Poisson rate).
    :param mu_J: Media del tamaño del salto logarítmico.
    :param sigma_J: Desviación del tamaño del salto logarítmico.
    :return: Matriz de precios simulados.
    """
    dt = T / N
    
    # 1. Componente de Difusión (MBG)
    dW = norm.rvs(size=(N, M), scale=np.sqrt(dt)) 
    
    # 2. Componente de Salto (Poisson)
    # Generar el número de saltos en cada paso de tiempo
    dN = poisson.rvs(lambda_p * dt, size=(N, M))
    # Generar el tamaño del salto (log-normal)
    dJ = norm.rvs(loc=mu_J, scale=sigma_J, size=(N, M))
    
    # 3. Corrección de la deriva (para mantener riesgo-neutral)
    # E[J] = exp(mu_J + 0.5 * sigma_J^2) - 1. Aquí usamos E[log(1+J)] = mu_J
    # Factor de corrección: gamma = mu - lambda * (exp(mu_J + 0.5 * sigma_J^2) - 1)
    
    gamma = mu - lambda_p * (np.exp(mu_J + 0.5 * sigma_J**2) - 1)
    
    S = np.zeros((N + 1, M))
    S[0] = S0
    
    for t in range(1, N + 1):
        # Componente de Difusión y Deriva
        diffusion_term = (gamma - 0.5 * sigma**2) * dt + sigma * dW[t-1]
        
        # Componente de Salto
        jump_term = np.where(dN[t-1] > 0, np.sum(norm.rvs(loc=mu_J, scale=sigma_J, size=(dN[t-1].max(), M)), axis=0), 0.0)
        
        # S[t] = S[t-1] * exp(Difusión + Salto)
        S[t] = S[t-1] * np.exp(diffusion_term + jump_term)
        
    return S

# ------------------------------------------------------------------------------
# CALIBRACIÓN Y SIMULACIÓN DE HESTON (VOLATILIDAD ESTOCÁSTICA)
# ------------------------------------------------------------------------------

def calibration_heston(params, log_returns):
    """Función objetivo simple para la calibración de Heston.
    La calibración real requiere métodos complejos de Fourier/MCL."""
    
    # Parámetros a estimar: [kappa, theta, xi, rho]
    # Asumimos que r y v0 (volatilidad inicial) vienen del MBG.
    kappa, theta, xi, rho = params
    
    # Para la simpleza, devolveremos un valor grande para evitar optimización inestable
    # En un entorno real, se usaría un método de optimización numérica sobre
    # la función de densidad o precios de opciones (Opciones de VIX o VIX en sí).
    
    # Usamos la varianza de la varianza como proxy
    vol_of_vol = log_returns.rolling(window=20).std().dropna().std()
    
    # La varianza teórica del modelo Heston para el precio de la opción no es simple.
    # Aquí solo calibraremos la varianza de la volatilidad.
    # Objetivo: theta (varianza a largo plazo) se acerque a la varianza histórica media
    # y kappa (velocidad de reversión) sea positiva.
    
    target_var = log_returns.var() * 252 # Varianza anual
    
    # Penalización por alejarse de la varianza histórica y por parámetros no válidos
    penalty = (theta - target_var)**2 + (1 - rho**2)**2 * 1000 + (xi < 0.001) * 1000
    
    return penalty

def calibrate_heston_simple(log_returns, sigma_daily):
    """Calibración simple de Heston (valores iniciales razonables)."""
    
    # Volatilidad inicial v0 (varianza diaria)
    v0 = sigma_daily**2 
    
    # Varianza media anual de los retornos (Theta - Varianza a largo plazo)
    theta_initial = log_returns.var() * 252
    
    # Valores iniciales: [kappa (Velocidad de reversión), theta (Varianza a largo plazo), xi (Vol de vol), rho (Correlación)]
    # kappa debe ser positivo y rho entre -1 y 1
    initial_params = [2.0, theta_initial, 0.2, -0.7]
    
    # La calibración real es compleja. Solo devolveremos los valores iniciales ajustados
    # res = minimize(calibration_heston, initial_params, args=(log_returns,), method='L-BFGS-B', bounds=[(0.01, 5), (0.0001, 0.5), (0.001, 1.0), (-0.99, 0.99)])
    # kappa, theta, xi, rho = res.x
    
    kappa, theta, xi, rho = initial_params
    
    print(f"\n--- Parámetros Calibrados de Heston (Iniciales Simples) ---")
    print(f"Varianza Inicial (v0): {v0:.6f}")
    print(f"Velocidad de reversión (kappa): {kappa:.4f}")
    print(f"Varianza a largo plazo (theta): {theta:.6f}")
    print(f"Volatilidad de la volatilidad (xi): {xi:.4f}")
    print(f"Correlación (rho): {rho:.4f}")
    print(f"----------------------------------------------------------")
    
    return v0, kappa, theta, xi, rho

def simulate_heston(S0, v0, kappa, theta, xi, rho, T, N, M=1, mu=0.0):
    """
    Simulación del Modelo de Heston (Volatilidad Estocástica).
    
    Ecuación de precios: dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW1_t
    Ecuación de volatilidad: dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
    donde dW1_t y dW2_t están correlacionados con rho.
    
    :param v0: Varianza inicial (volatilidad^2).
    :return: Matriz de precios simulados.
    """
    dt = T / N
    
    # Generar números aleatorios correlacionados
    Z1 = norm.rvs(size=(N, M))
    Z2 = norm.rvs(size=(N, M))
    
    dW1 = np.sqrt(dt) * Z1 # Movimiento Browniano para el precio
    dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2) # Movimiento Browniano para la varianza
    
    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    
    S[0] = S0
    v[0] = v0
    
    for t in range(1, N + 1):
        # 1. Simulación de la Varianza (Método de Euler con Absorbing/Reflection)
        # Aseguramos que la varianza no sea negativa (Trunca a 0 si es < 0)
        dv_t = kappa * (theta - v[t-1]) * dt + xi * np.sqrt(v[t-1]) * dW2[t-1]
        v[t] = v[t-1] + dv_t
        v[t] = np.maximum(v[t], 0) # Reflection (o Absorbing) Boundary
        
        # 2. Simulación del Precio (usando la nueva varianza)
        ds_t = mu * S[t-1] * dt + np.sqrt(v[t-1]) * S[t-1] * dW1[t-1]
        S[t] = S[t-1] + ds_t
        
    return S

# ==============================================================================
# 4. BACKTESTING Y MÉTRICAS
# ==============================================================================

def calculate_rmse(historical_prices, simulated_paths):
    """
    Calcula el Error Cuadrático Medio (RMSE) entre los precios históricos
    y la media de las trayectorias simuladas.
    
    :param historical_prices: Precios observados (Serie de pandas).
    :param simulated_paths: Matriz de precios simulados (N+1 x M).
    :return: RMSE.
    """
    # 1. Asegurar que las longitudes coincidan
    # La simulación tiene (N+1) puntos. Quitamos el punto inicial para comparar N puntos.
    
    # Cortar los precios históricos para que coincidan con la longitud de la simulación
    N_sim = simulated_paths.shape[0] - 1 
    if len(historical_prices) < N_sim + 1:
        # Esto debería manejarse antes, pero como fallback, recortamos la simulación
        sim_mean = simulated_paths[:len(historical_prices)].mean(axis=1)
        print("Advertencia: Longitud histórica menor que la simulación.")
    else:
        # Usamos la ventana histórica que coincide con la simulación
        hist_window = historical_prices[-N_sim-1:]
        sim_mean = simulated_paths.mean(axis=1)

    # El RMSE se calcula solo para los precios futuros, no incluyendo S0
    rmse = np.sqrt(np.mean((hist_window.values[1:] - sim_mean[1:])**2))
    
    return rmse

# ==============================================================================
# 5. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================

def run_backtesting(ticker, start_date, end_date, simulation_days, num_simulations=1000):
    """
    Ejecuta el flujo completo de descarga, calibración, simulación y backtesting.
    """
    # 1. Descargar datos
    prices = fetch_data(ticker, start_date, end_date)
    if prices is None:
        return

    # 2. Dividir datos para calibración y backtesting
    # Usamos el precio final del set de calibración como S0
    S0_historical = prices.iloc[-simulation_days] # Precio inicial para la simulación
    
    # Datos de calibración
    calibration_prices = prices.iloc[:-simulation_days]
    calibration_log_returns = calculate_log_returns(calibration_prices)
    
    # Datos de backtesting (histórico real)
    backtest_prices = prices.iloc[-simulation_days-1:]
    
    if len(calibration_log_returns) < 50:
        print("Error: No hay suficientes datos para la calibración. Ajuste las fechas o los días de simulación.")
        return

    print(f"\n--- Backtesting de {ticker} ---")
    print(f"Periodo de Calibración: {calibration_prices.index[0].strftime('%Y-%m-%d')} a {calibration_prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"Día de Simulación (S0): {backtest_prices.index[0].strftime('%Y-%m-%d')} (Precio: ${S0_historical:.2f})")
    print(f"Días a Simular: {simulation_days} (Hasta {prices.index[-1].strftime('%Y-%m-%d')})")
    
    # 3. Calibrar parámetros básicos (MBG)
    mu, sigma, mu_daily, sigma_daily = get_basic_parameters(calibration_log_returns, dt=1/252)

    # ------------------------------------
    # Simulación y RMSE para MBG
    # ------------------------------------
    print("\n[Modelo 1: Movimiento Browniano Geométrico]")
    # Usamos mu_daily y sigma_daily como r y v para la simulación
    gbm_paths = simulate_gbm(S0_historical, mu_daily, sigma_daily, simulation_days/252, simulation_days, num_simulations)
    rmse_gbm = calculate_rmse(backtest_prices, gbm_paths)
    print(f"RMSE (MBG): {rmse_gbm:.4f}")

    # ------------------------------------
    # Calibración, Simulación y RMSE para MERTON
    # ------------------------------------
    # Calibrar Merton (saltos)
    lambda_p, mu_J, sigma_J = calibrate_merton_simple(calibration_log_returns)
    
    print("\n[Modelo 2: Salto-Difusión de Merton]")
    # Usamos mu_daily y sigma_daily como mu y sigma de difusión
    merton_paths = simulate_merton(S0_historical, mu_daily, sigma_daily, lambda_p, mu_J, sigma_J, simulation_days/252, simulation_days, num_simulations)
    rmse_merton = calculate_rmse(backtest_prices, merton_paths)
    print(f"RMSE (Merton): {rmse_merton:.4f}")

    # ------------------------------------
    # Calibración, Simulación y RMSE para HESTON
    # ------------------------------------
    # Calibrar Heston (volatilidad estocástica)
    v0, kappa, theta, xi, rho = calibrate_heston_simple(calibration_log_returns, sigma_daily)
    
    print("\n[Modelo 3: Volatilidad Estocástica de Heston]")
    # Usamos mu_daily como drift (r)
    heston_paths = simulate_heston(S0_historical, v0, kappa, theta, xi, rho, simulation_days/252, simulation_days, num_simulations, mu=mu_daily)
    rmse_heston = calculate_rmse(backtest_prices, heston_paths)
    print(f"RMSE (Heston): {rmse_heston:.4f}")
    
    # ------------------------------------
    # Resultado Final
    # ------------------------------------
    results = {
        'MBG': rmse_gbm,
        'Merton': rmse_merton,
        'Heston': rmse_heston
    }
    
    best_model = min(results, key=results.get)
    
    print(f"\n=======================================================")
    print(f"El mejor modelo (menor RMSE) para {ticker} es: {best_model}")
    print(f"=======================================================")
    
    # ------------------------------------
    # Preparar datos para la visualización en Streamlit
    # ------------------------------------
    
    # Crear un DataFrame con las trayectorias simuladas y el precio histórico real
    # para Streamlit. Solo la media de la simulación.
    
    dates = backtest_prices.index[1:] # Fechas simuladas (excluyendo S0)
    
    # Asegurar que las longitudes coincidan (quitando S0)
    data_for_streamlit = pd.DataFrame({
        'Fecha': dates,
        f'{ticker} Real': backtest_prices.values[1:],
        'MBG Simulado (Media)': gbm_paths.mean(axis=1)[1:],
        'Merton Simulado (Media)': merton_paths.mean(axis=1)[1:],
        'Heston Simulado (Media)': heston_paths.mean(axis=1)[1:],
    }).set_index('Fecha')
    
    print("\nDataFrame de resultados para Streamlit generado.")
    # El archivo debe guardarse o serializarse para Streamlit.
    # Usaremos una variable global/return para este ejemplo.
    return results, data_for_streamlit


if __name__ == '__main__':
    # --- CONFIGURACIÓN DEL USUARIO ---
    
    # Ticker que deseas analizar (ej: 'GOOG', 'TSLA', '^GSPC')
    TICKER = 'AAPL' 
    
    # Rango de fechas para OBTENER todos los datos (Calibración + Simulación)
    # Se recomienda un rango de al menos 1-2 años para calibrar bien.
    START_DATE = '2023-01-01' 
    END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    # Número de días hábiles que deseas simular/backtestear (ej: los últimos 30 días)
    # Los datos serán divididos: el resto para calibración y esta cantidad para simulación.
    SIMULATION_DAYS = 30 
    
    # Número de trayectorias de Monte Carlo
    NUM_SIMULATIONS = 1000 
    
    # --- EJECUCIÓN ---
    
    results, data_for_streamlit = run_backtesting(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        simulation_days=SIMULATION_DAYS,
        num_simulations=NUM_SIMULATIONS
    )
    
    if data_for_streamlit is not None:
        print("\n--- Vista previa del DataFrame para Streamlit ---")
        print(data_for_streamlit.tail())
        # Aquí es donde guardarías el archivo para tu app de Streamlit (ej: .csv, .json)
        # data_for_streamlit.to_csv('simulacion_resultados.csv')
