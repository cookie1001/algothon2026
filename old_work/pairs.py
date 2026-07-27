import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import coint

def backtest_all_pairs(df, lookback=20, entry_z=2.0, capital=10000, risk_free_rate=0.0):
    """
    Tests all combinations of assets in the provided DataFrame for a Pairs Trading strategy.
    
    Parameters:
    - df: Pandas DataFrame where index is dates and columns are asset tickers with daily closing prices.
    - lookback: Rolling window for spread mean and standard deviation.
    - entry_z: Z-score threshold to trigger a trade.
    - capital: Starting capital per pair.
    - risk_free_rate: Assumed daily risk-free rate for Sharpe ratio calculation.
    """
    results = []
    
    # 1. Generate all unique asset pairs from your 51 columns
    tickers = df.columns
    all_pairs = list(itertools.combinations(tickers, 2))
    print(f"Testing {len(all_pairs)} possible pairs...")
    
    # 2. Iterate through every combination
    for asset_A, asset_B in all_pairs:
        # Extract series and drop missing values to align dates
        price_A = df[asset_A]
        price_B = df[asset_B]
        
        # Ensure we have enough data to test
        if len(price_A.dropna()) < lookback * 2:
            continue
            
        # 3. Engle-Granger Cointegration Test (Single Function)
        # We only want to trade pairs that historically revert to the mean
        t_stat, p_value, _ = coint(price_A.dropna(), price_B.dropna())
        
        # If p_value > 0.05, we cannot reject the null hypothesis; spread is not stationary
        if p_value > 0.05:
            continue 
            
        # 4. Strategy Execution
        pair_df = pd.DataFrame({'Asset_A': price_A, 'Asset_B': price_B}).dropna()
        
        # Calculate the spread
        pair_df['Spread'] = pair_df['Asset_A'] - pair_df['Asset_B']
        
        # Calculate Rolling Z-Score
        pair_df['Rolling_Mean'] = pair_df['Spread'].rolling(window=lookback).mean()
        pair_df['Rolling_Std'] = pair_df['Spread'].rolling(window=lookback).std()
        pair_df['Z_Score'] = (pair_df['Spread'] - pair_df['Rolling_Mean']) / pair_df['Rolling_Std']
        
        # Generate Trading Signals
        pair_df['Position'] = 0
        pair_df.loc[pair_df['Z_Score'] < -entry_z, 'Position'] = 1   # Buy spread
        pair_df.loc[pair_df['Z_Score'] > entry_z, 'Position'] = -1  # Short spread
        
        # Forward fill and exit on mean reversion (Z-score crossing 0)
        pair_df['Position'] = pair_df['Position'].replace(0, method='ffill')
        pair_df.loc[(pair_df['Position'] == 1) & (pair_df['Z_Score'] >= 0), 'Position'] = 0
        pair_df.loc[(pair_df['Position'] == -1) & (pair_df['Z_Score'] <= 0), 'Position'] = 0
        
        # Calculate Returns
        pair_df['Asset_A_Ret'] = pair_df['Asset_A'].pct_change()
        pair_df['Asset_B_Ret'] = pair_df['Asset_B'].pct_change()
        
        # Shift position by 1 to prevent look-ahead bias
        pair_df['Strategy_Ret'] = pair_df['Position'].shift(1) * (pair_df['Asset_A_Ret'] - pair_df['Asset_B_Ret'])
        
        # 5. Performance Metrics
        pair_df = pair_df.dropna()
        if len(pair_df) == 0:
            continue
            
        total_return = (1 + pair_df['Strategy_Ret']).prod() - 1
        total_profit = capital * total_return
        
        # Annualized Sharpe Ratio (assuming 252 daily trading days)
        excess_returns = pair_df['Strategy_Ret'] - risk_free_rate
        if excess_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        else:
            sharpe_ratio = 0.0
            
        results.append({
            'Asset_A': asset_A,
            'Asset_B': asset_B,
            'Cointegration_P_Value': round(p_value, 4),
            'Total_Profit_USD': round(total_profit, 2),
            'Sharpe_Ratio': round(sharpe_ratio, 3)
        })
        
    # 6. Format and sort results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Sort by best Sharpe Ratio
        results_df = results_df.sort_values(by='Sharpe_Ratio', ascending=False).reset_index(drop=True)
        
    return results_df

# ==========================================
# HOW TO RUN IT WITH YOUR DATASET
# ==========================================

# 1. Load your dataset (assuming it is a CSV file)
# Replace 'your_price_matrix.csv' with your actual file path
# df = pd.read_csv('your_price_matrix.csv', index_col=0, parse_dates=True)
df = pd.read_csv('prices.txt', delim_whitespace=True)
best_pairs_df = backtest_all_pairs(df)
print(best_pairs_df.head(10))

# 2. Run the backtest function
# best_pairs_df = backtest_all_pairs(df)

# 3. View the top 10 most profitable/efficient pairs
# print(best_pairs_df.head(10))