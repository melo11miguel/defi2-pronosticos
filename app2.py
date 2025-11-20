import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Modelado Financiero y Backtesting")

# --- Funciones de Modelos y Simulación ---

def estimate_gbm_parameters(log_returns):
    """Estima la deriva (mu) y la volatilidad (sigma) para GBM."""
    mu = log_returns.mean()
    sigma = log_returns.std()
    return mu, sigma

def simulate_gbm(S0, mu, sigma, T, N_paths, N_steps):
    """Simulación de Movimiento Browniano Geométrico."""
    # Corrección de seguridad: Aseguramos que los parámetros de entrada sean escalares (float)
    S0 = float(S0)
    mu = float(mu)
    sigma = float(sigma)
    
    dt = T / N_steps
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0
    
    for t in range(1, N_steps + 1):
        # Generar movimientos brownianos
        dW = np.random.normal(0, np.sqrt(dt), N_paths)
        # La operación es puramente NumPy
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
    return paths

def simulate_heston(S0, V0, mu, kappa, theta, sigma_v, rho, T, N_paths, N_steps):
    """Simulación del Modelo de Heston (volatilidad estocástica)."""
    S0 = float(S0)
    mu = float(mu)
    
    dt = T / N_steps
    paths = np.zeros((N_steps + 1, N_paths))
    v_paths = np.zeros((N_steps + 1, N_paths))
    
    paths[0] = S0
    v_paths[0] = V0
    
    for t in range(1, N_steps + 1):
        # Generar números aleatorios correlacionados
        z1 = np.random.normal(0.0, 1.0, N_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0.0, 1.0, N_paths)
        
        # Simulación de la volatilidad (CIR Process - Euler-Maruyama)
        dv = kappa * (theta - v_paths[t-1]) * dt + sigma_v * np.sqrt(v_paths[t-1]) * np.sqrt(dt) * z2
        v_paths[t] = v_paths[t-1] + dv
        
        # Asegurar que la volatilidad no sea negativa (Full Truncation Scheme)
        v_paths[t] = np.maximum(v_paths[t], 1e-6)
        
        # Simulación del precio de la acción
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * v_paths[t-1]) * dt + np.sqrt(v_paths[t-1]) * np.sqrt(dt) * z1)
        
    return paths

def simulate_merton(S0, mu, sigma, lambda_j, m, v, T, N_paths, N_steps):
    """Simulación del Modelo de Merton (Salto-Difusión)."""
    S0 = float(S0)
    mu = float(mu)
    sigma = float(sigma)
    
    dt = T / N_steps
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0
    
    # Parámetro de compensación para la media
    gamma = mu - 0.5 * sigma**2 - lambda_j * (np.exp(m + 0.5 * v**2) - 1)
    
    for t in range(1, N_steps + 1):
        # Generar movimientos brownianos (dW)
        dW = np.random.normal(0, np.sqrt(dt), N_paths)
        
        # Generar el número de saltos (Poisson Process - dN)
        dN = np.random.poisson(lambda_j * dt, N_paths)
        
        # Generar el tamaño del salto (Jump size - dJ)
        dJ = np.where(dN > 0, np.random.normal(m, v, N_paths), 0.0)
        
        # Sumar todos los componentes
        log_return = gamma * dt + sigma * dW + dJ
        paths[t] = paths[t-1] * np.exp(log_return)
        
    return paths


def backtest_model(df_prices, model_func, model_name, N_paths, T_test_days, **params):
    """
    Realiza el backtesting y calcula el RMSE.
    """
    
    # 1. Definir el conjunto de entrenamiento (Train) y prueba (Test)
    train_prices = df_prices.iloc[:-T_test_days]
    test_prices = df_prices.iloc[-T_test_days:]
    
    # Convertir S0 a un flotante puro (escalar)
    S0 = float(train_prices.iloc[-1])
    
    N_steps = T_test_days
    T = T_test_days / 252.0  # Asumimos 252 días de trading al año
    
    # 2. Simular el modelo
    if model_name == "GBM":
        log_returns = np.log(train_prices / train_prices.shift(1)).dropna()
        mu, sigma = estimate_gbm_parameters(log_returns)
        simulated_paths = model_func(S0, mu, sigma, T, N_paths, N_steps)
    elif model_name == "Heston":
        log_returns = np.log(train_prices / train_prices.shift(1)).dropna()
        mu, sigma = estimate_gbm_parameters(log_returns) 
        
        mu_annualized = float(mu) * 252
        v0_initial = float(sigma)**2 * 252
        
        simulated_paths = model_func(
            S0=S0, 
            mu=mu_annualized,
            V0=v0_initial,
            T=T, 
            N_paths=N_paths, 
            N_steps=N_steps,
            kappa=params['kappa'], 
            theta=params['theta'], 
            sigma_v=params['sigma_v'], 
            rho=params['rho']
        )
    elif model_name == "Merton":
        log_returns = np.log(train_prices / train_prices.shift(1)).dropna()
        mu, sigma = estimate_gbm_parameters(log_returns)
        
        mu_annualized = float(mu) * 252
        sigma_annualized = float(sigma) * np.sqrt(252)
        
        simulated_paths = model_func(
            S0=S0, 
            mu=mu_annualized,
            sigma=sigma_annualized,
            T=T, 
            N_paths=N_paths, 
            N_steps=N_steps,
            lambda_j=params['lambda_j'],
            m=params['m'],
            v=params['v']
        )
    else:
        st.error(f"Modelo {model_name} no reconocido.")
        return 0, pd.DataFrame()
        
    # 3. Calcular el promedio de las simulaciones
    simulated_mean = np.mean(simulated_paths, axis=1)
    
    # Asegurarse de que las longitudes coincidan
    
    # CORRECCIÓN CLAVE: Usamos .values.flatten() para garantizar que sea 1D
    actual_prices = test_prices.values.flatten()
    predicted_prices = simulated_mean[1:] 
    
    if len(actual_prices) != len(predicted_prices):
        min_len = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:min_len]
        predicted_prices = predicted_prices[:min_len]
        
    # 4. Calcular el RMSE
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    
    # 5. Crear DataFrame de resultados para visualización
    results_df = pd.DataFrame({
        'Fecha': test_prices.index,
        'Precio Real': actual_prices,
        f'Precio Promedio Simulado ({model_name})': predicted_prices
    }).set_index('Fecha')
    
    return rmse, results_df


# --- Función Principal de Streamlit (CON LÓGICA DE RESPALDO PARA 'Close') ---

def main():
    st.title("💸 Modelado de Precios de Activos y Backtesting")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Configuración de la Aplicación")
        
        # 1. Entrada de Ticker
        ticker = st.text_input("Ingrese el Símbolo del Activo (Ticker)", "AAPL").upper()
        
        # 2. Configuración de Datos
        start_date = st.date_input("Fecha de Inicio de Datos", pd.to_datetime("2020-01-01"))
        # Usamos yesterday como sugerencia predeterminada
        yesterday = pd.to_datetime("today") - pd.Timedelta(days=1)
        end_date = st.date_input("Fecha de Fin de Datos", yesterday)
        
        # 3. Configuración del Backtesting
        st.subheader("Configuración de Simulación")
        test_days = st.slider("Días a Predecir (Backtest)", 10, 60, 21)
        n_paths = st.slider("Número de Trayectorias de Simulación", 100, 1000, 300)
        
        # 4. Parámetros de Heston (Para demostración)
        st.subheader("Parámetros de Heston (Aprox.)")
        st.caption("La calibración precisa requiere datos de opciones.")
        kappa = st.number_input("Tasa de Reversión (kappa)", 0.01, 5.0, 1.5, 0.1)
        theta = st.number_input("Varianza de Largo Plazo (theta)", 0.01, 1.0, 0.05, 0.01)
        sigma_v = st.number_input("Volatilidad de la Varianza (sigma_v)", 0.01, 5.0, 0.2, 0.05)
        rho = st.number_input("Correlación (rho)", -1.0, 1.0, -0.7, 0.1)
        
        # 5. Parámetros de Merton (Para demostración)
        st.subheader("Parámetros de Merton (Aprox.)")
        st.caption("La calibración precisa requiere inferencia estadística avanzada.")
        lambda_j = st.number_input("Frecuencia de Salto (lambda_j)", 0.01, 1.0, 0.2, 0.05)
        m = st.number_input("Tamaño Promedio de Salto (m)", -0.5, 0.5, -0.05, 0.01)
        v = st.number_input("Volatilidad del Salto (v)", 0.01, 0.5, 0.1, 0.01)


    with col2:
        if st.button(f"Ejecutar Backtesting para {ticker}", type="primary"):
            
            # --- 1. Descarga de Datos ---
            st.subheader(f"1. Descarga de Datos: {ticker}")
            
            try:
                # Descargar datos
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                # Verificar si el DataFrame está vacío
                if data.empty:
                    st.error(f"No se encontraron datos históricos para el ticker {ticker} entre {start_date} y {end_date}. Intente con un rango de fechas diferente o un ticker válido.")
                    return

                # Lógica de Respaldo: Usar 'Close' si 'Adj Close' no existe
                if 'Adj Close' in data.columns:
                    df_prices = data['Adj Close'].dropna()
                    columna_usada = 'Cierre Ajustado (Adj Close)'
                elif 'Close' in data.columns:
                    df_prices = data['Close'].dropna()
                    columna_usada = 'Cierre (Close)'
                else:
                    st.error("Error: No se encontró la columna 'Adj Close' ni 'Close' en los datos descargados. Verifique el ticker (símbolo).")
                    return


                if df_prices.empty:
                    st.error(f"Error: La columna {columna_usada} está vacía después de limpiar valores nulos. Revise las fechas y el ticker.")
                    return

                st.write(f"Primeras 5 filas de precios, usando: **{columna_usada}**")
                st.dataframe(df_prices.head())
                
                # Definir conjuntos de entrenamiento y prueba para backtesting
                if len(df_prices) <= test_days:
                    st.error(f"Error: No hay suficientes datos ({len(df_prices)} puntos) para realizar un backtesting de {test_days} días.")
                    return
                
                train_prices = df_prices.iloc[:-test_days]
                test_prices = df_prices.iloc[-test_days:]

            except Exception as e:
                st.error(f"Error al descargar datos para {ticker}: {e}")
                return

            # --- 2. Ejecutar Backtesting para los 3 Modelos ---
            st.subheader("2. Resultados del Backtesting y RMSE")
            
            # --- GBM ---
            rmse_gbm, results_gbm = backtest_model(
                df_prices=df_prices, 
                model_func=simulate_gbm, 
                model_name="GBM", 
                N_paths=n_paths, 
                T_test_days=test_days
            )
            
            # --- Heston ---
            heston_params = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho}
            rmse_heston, results_heston = backtest_model(
                df_prices=df_prices, 
                model_func=simulate_heston, 
                model_name="Heston", 
                N_paths=n_paths, 
                T_test_days=test_days,
                **heston_params
            )

            # --- Merton ---
            merton_params = {'lambda_j': lambda_j, 'm': m, 'v': v}
            rmse_merton, results_merton = backtest_model(
                df_prices=df_prices, 
                model_func=simulate_merton, 
                model_name="Merton", 
                N_paths=n_paths, 
                T_test_days=test_days,
                **merton_params
            )
            
            # --- 3. Consolidar Resultados ---
            
            # DataFrame de RMSE
            rmse_df = pd.DataFrame({
                'Modelo': ['Geométrico Browniano (GBM)', 'Heston', 'Merton'],
                'RMSE': [rmse_gbm, rmse_heston, rmse_merton]
            })
            
            # Identificar el mejor modelo
            best_model = rmse_df.loc[rmse_df['RMSE'].idxmin()]
            
            st.markdown("### Tabla de Comparación de RMSE")
            st.dataframe(rmse_df.set_index('Modelo').style.highlight_min(axis=0, color='lightgreen'))

            st.markdown(f"#### 🎉 El Mejor Modelo (menor RMSE) es: **{best_model['Modelo']}** con RMSE de **{best_model['RMSE']:.4f}**")

            st.markdown("---")
            st.subheader("4. Visualización de la Predicción (Backtest)")
            
            # Combinar los resultados simulados y reales
            combined_df = results_gbm.join(results_heston.iloc[:, 1]).join(results_merton.iloc[:, 1])
            combined_df.columns = ['Precio Real', 'GBM', 'Heston', 'Merton']
            
            # Preparar datos de entrenamiento para el gráfico
            train_plot = train_prices.to_frame(name='Precio')
            
            # Agregar los datos simulados
            plot_df = pd.concat([train_plot, combined_df], axis=0)
            
            # Crear un gráfico interactivo
            st.line_chart(plot_df, height=500)
            st.caption(f"El Backtest compara el precio real vs. el precio promedio simulado para los últimos {test_days} días.")

if __name__ == "__main__":
    main()
