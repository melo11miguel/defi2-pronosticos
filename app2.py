# archivo: options_backtest.py
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import math
import datetime

# -------- Black-Scholes (call & put) -----------
def bs_price(S, K, r, sigma, tau, option_type='call'):
    if tau <= 0:
        return max(0.0, (S-K) if option_type=='call' else (K-S))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    if option_type=='call':
        return S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    else:
        return K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)

# implied volatility (inverts market price to sigma)
def implied_vol(price, S, K, r, tau, option_type='call', sigma_bounds=(1e-6, 5.0)):
    f = lambda sigma: bs_price(S,K,r,sigma,tau,option_type)-price
    try:
        iv = brentq(f, sigma_bounds[0], sigma_bounds[1], maxiter=200)
        return iv
    except Exception:
        return np.nan

# -------- Binomial CRR (European) -----------
def binomial_crr_price(S, K, r, sigma, tau, steps=100, option_type='call'):
    dt = tau/steps
    u = math.exp(sigma*math.sqrt(dt))
    d = 1.0/u
    disc = math.exp(-r*dt)
    p = (math.exp(r*dt)-d)/(u-d)
    # terminal prices
    prices = np.array([S * (u**j) * (d**(steps-j)) for j in range(steps+1)])
    if option_type=='call':
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)
    # backward induction
    for i in range(steps-1, -1, -1):
        values = disc*(p*values[1:] + (1-p)*values[:-1])
    return values[0]

# -------- Monte Carlo (GBM) -----------
def mc_price_gbm(S, K, r, sigma, tau, option_type='call', n_sims=20000):
    Z = np.random.normal(size=n_sims)
    ST = S * np.exp((r - 0.5*sigma**2)*tau + sigma*np.sqrt(tau)*Z)
    if option_type=='call':
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    return np.exp(-r*tau) * payoffs.mean()

# -------- Backtest skeleton -----------
def backtest_one_day(ticker, options_history, models=['bs','binomial','mc'],
                     train_window_days=252, start_date=None, end_date=None, r=0.01):
    # options_history: DataFrame with columns:
    # date, expiry (datetime), strike, option_type ('call'/'put'), mid_price, underlying_close
    # We'll iterate day by day: for each day t we predict t+1 price for same contract.
    # ---- prepare date range
    if start_date is None:
        start_date = options_history['date'].min() + pd.Timedelta(days=train_window_days)
    if end_date is None:
        end_date = options_history['date'].max()-pd.Timedelta(days=1)
    dates = pd.date_range(start_date, end_date, freq='B')
    results = []
    for t in dates:
        # subset the universe: choose contracts that exist at t and have price at t+1
        today = t
        tomorrow = t + pd.Timedelta(days=1)
        todays_contracts = options_history[options_history['date']==today]
        if todays_contracts.empty:
            continue
        for _, row in todays_contracts.iterrows():
            # find observed price at t+1
            obs = options_history[
                (options_history['date']==tomorrow) &
                (options_history['strike']==row['strike']) &
                (options_history['expiry']==row['expiry']) &
                (options_history['option_type']==row['option_type'])
            ]
            if obs.empty:
                continue
            obs_price = obs['mid_price'].values[0]
            # prepare inputs
            S_t = row['underlying_close']
            K = row['strike']
            tau_t = (row['expiry'] - today).days / 365.0
            tau_t1 = (row['expiry'] - tomorrow).days / 365.0
            # estimate sigma_t (historic vol over train_window_days of underlying up to t)
            # For simplicity: use daily log returns std * sqrt(252)
            past_prices = options_history[(options_history['date']<=today)]['underlying_close'].drop_duplicates().tail(train_window_days)
            if len(past_prices) < 10:
                continue
            logrets = np.log(past_prices).diff().dropna()
            sigma_hist = logrets.std()*math.sqrt(252)
            # Predicted S_{t+1} (simple forecast): random walk with drift 0 -> S_t (or use AR model)
            S_pred = S_t  # naive (can be improved)
            # now compute predictions per model
            preds = {}
            if 'bs' in models:
                # Use sigma_hist as proxy for sigma_{t+1}
                preds['bs'] = bs_price(S_pred, K, r, sigma_hist, tau_t1, row['option_type'])
            if 'binomial' in models:
                preds['binomial'] = binomial_crr_price(S_pred, K, r, sigma_hist, tau_t1, steps=50, option_type=row['option_type'])
            if 'mc' in models:
                preds['mc'] = mc_price_gbm(S_pred, K, r, sigma_hist, tau_t1, option_type=row['option_type'], n_sims=5000)
            # record
            for m, p in preds.items():
                results.append({
                    'date': today,
                    'ticker': ticker,
                    'strike': K,
                    'expiry': row['expiry'],
                    'option_type': row['option_type'],
                    'model': m,
                    'pred_price': float(p),
                    'obs_price': float(obs_price),
                    'error': float(p - obs_price)
                })
    df_results = pd.DataFrame(results)
    # compute RMSE per model
    rmse = df_results.groupby('model').apply(lambda g: np.sqrt(np.mean((g['pred_price']-g['obs_price'])**2)))
    return df_results, rmse

# -------- Example: download underlying 2 years -----------
def fetch_underlying(ticker, period_years=2):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365*period_years)
    df = yf.download(ticker, start=start, end=end)
    df = df[['Adj Close']].rename(columns={'Adj Close':'close'}).reset_index()
    df['date'] = pd.to_datetime(df['Date']).dt.date
    df['close'] = df['close'].astype(float)
    return df[['date','close']]
