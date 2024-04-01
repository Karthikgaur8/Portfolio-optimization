
# Portfolio Optimization Tool

A Python-based application designed to optimize investment portfolios by maximizing the Sharpe Ratio through the integration of historical stock data analysis, machine learning predictions, and advanced mathematical optimization techniques.

## Project Overview

This tool automates the process of creating an optimized stock portfolio that seeks to balance risk and reward more efficiently than traditional methods. By leveraging historical data and predictive models, it provides investors with insights to enhance portfolio performance under various market conditions.

## Core Features

### Data Acquisition
- **Historical Market Data**: Utilizes `yfinance` to fetch historical price data for a predefined set of stocks over a specific period. This data forms the foundation for further analysis and model training.
- **Risk-Free Rate Retrieval**: Incorporates current risk-free rate data from the Federal Reserve Economic Data (FRED) using the `fredapi`, critical for Sharpe Ratio calculations.

### Predictive Modeling for Future Returns
- **Linear Regression Model**: Implements a linear regression model to forecast future stock returns based on historical data. This model helps in understanding potential future performance and enhances the decision-making process for portfolio composition.
- **Mean Squared Error (MSE) Evaluation**: Evaluates the predictive model's accuracy by calculating the MSE between the model's predictions and the actual returns, providing a quantitative measure of the model's performance.

### Portfolio Optimization
- **Sharpe Ratio Maximization**: Uses the `scipy.optimize` library to find the portfolio weights that maximize the Sharpe Ratio, indicating an optimal risk-adjusted return profile.
- **Covariance Matrix Analysis**: Calculates the covariance matrix from log returns to assess the volatility and correlation between stocks, aiding in the diversification strategy.

### Visualization and Analytics
- **Optimal Weights Visualization**: Presents a bar chart of the optimized portfolio weights using `matplotlib`, offering a clear visual representation of the investment strategy.
- **Performance Metrics Display**: Outputs key portfolio metrics, including expected annual return, portfolio volatility, and the Sharpe Ratio, to evaluate the optimized portfolio's performance.

## Technologies Used

- **Python**: The core programming language used for developing the tool.
- **Pandas & NumPy**: Essential for efficient data manipulation and numerical operations.
- **Matplotlib**: For generating visualizations of the portfolio weights and potentially other analytics.
- **SciPy**: Provides functions for mathematical optimization.
- **Scikit-learn**: Facilitates the construction and evaluation of the machine learning model.
- **yfinance**: Enables access to Yahoo Finance's historical data.
- **FRED API**: For fetching the current risk-free rate, crucial in financial modeling.

