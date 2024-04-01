# Portfolio Optimization in Python

# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from fredapi import Fred

# Define stock tickers and the time range for historical data
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI', 'AAPL', 'GOOG', 'MSFT', 'TSLA']
start_date = datetime.today() - timedelta(days=10*365)
end_date = datetime.today()
print(f"Start date: {start_date}")

# Download adjusted close prices for the tickers
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Adj Close']

# Calculate daily log returns and drop missing values
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# Calculate the covariance matrix of log returns
cov_matrix = log_returns.cov() * 252

# Linear Regression Model for Predicting Future Returns
X = log_returns.shift(1).iloc[1:]  # Use previous day's returns as features
y = log_returns.iloc[1:]  # Today's returns as target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Prepare the dataset for future predictions (exclude the NaN introduced by shifting)
X_future = log_returns.shift(1).iloc[1:]  # This also aligns with the structure of your training data

# Predict future returns
future_returns = model.predict(X_future)

# Calculate expected future returns for optimization
# Ensure to handle dimensions correctly, especially if model.predict() returns a 2D array
expected_future_returns = np.mean(future_returns, axis=0)

# Portfolio Performance Metrics Functions
def standard_deviation(weights, cov_matrix):
    """Calculate portfolio standard deviation."""
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

def expected_return(weights, predicted_returns=expected_future_returns):
    """Calculate expected portfolio return."""
    return np.dot(predicted_returns, weights) * 252

def sharpe_ratio(weights, predicted_returns, cov_matrix, risk_free_rate):
    """Calculate the Sharpe Ratio of the portfolio."""
    return (expected_return(weights, predicted_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# Fetch the risk-free rate from FRED
fred = Fred(api_key='5c2f6eb43a7d2d00b90450e3e60af4ec')
risk_free_rate = fred.get_series_latest_release('GS10').iloc[-1] / 100

# Negative Sharpe Ratio (for minimization)
def neg_sharpe_ratio(weights, predicted_returns, cov_matrix, risk_free_rate):
    """Returns the negative of the Sharpe ratio."""
    return -sharpe_ratio(weights, predicted_returns, cov_matrix, risk_free_rate)

# Constraints and bounds for optimization
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 0.5) for _ in range(len(tickers)))
initial_weights = np.array([1/len(tickers)] * len(tickers))

# Optimize portfolio to maximize the Sharpe Ratio
optimized_result = minimize(neg_sharpe_ratio, initial_weights, args=(expected_future_returns, cov_matrix, risk_free_rate),
                            method='SLSQP', bounds=bounds, constraints=constraints)

# Display optimized weights and portfolio stats
optimal_weights = optimized_result.x
print("Optimal Weights:\n", optimal_weights)

# Display the portfolio performance
optimal_portfolio_return = expected_return(optimal_weights, expected_future_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = -optimized_result.fun  # Since we minimized the negative Sharpe ratio


#display analytics:
print(optimal_weights)
for ticker, weight in  zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")


print(f"\nExpected Annual Return: {optimal_portfolio_return:.4%}")
print(f"Portfolio Volatility: {optimal_portfolio_volatility:.4%}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

# Visualize the optimal portfolio weights
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights, color='skyblue')
plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')
plt.xticks(rotation=45)
plt.show()
