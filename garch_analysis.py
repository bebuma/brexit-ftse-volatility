import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

absolute_path = os.path.abspath(__file__)
output_folder = os.path.dirname(absolute_path) + "/output_garch/"

def retrieve_stock_data(symbol, start_date, end_date):
    """Retrieve stock data from Yahoo Finance."""
    return yf.download(symbol, start=start_date, end=end_date)

def calculate_returns(closing_prices):
    """Calculate daily returns from closing prices."""
    return 100 * closing_prices.pct_change().dropna()

def plot_stock_data(data, title, filename):
    """Plot and save stock data."""
    plt.figure(figsize=(15, 5))
    plt.plot(data)
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_pacf_squared_returns(returns, filename):
    """Generate and save PACF plot of squared returns."""
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_pacf(returns**2, method="ywm", ax=ax)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def fit_garch_model(returns, p, q):
    """Fit a GARCH model to returns and return the result."""
    model = arch_model(returns, p=p, q=q, dist='skewt', vol="GARCH")
    result = model.fit(disp='off')
    return result

def plot_garch_results(result, filename):
    """Plot and save GARCH model results."""
    fig = result.plot()
    fig.set_size_inches(15, 5)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_real_vs_predicted_returns(returns, conditional_volatility, filename):
    """Plot and save real returns vs. conditional volatility from GARCH result."""
    plt.figure(figsize=(15, 5))
    plt.plot(returns, label="Real Returns")
    plt.plot(conditional_volatility, label="Conditional Volatility")
    plt.title("GARCH Model")
    plt.legend(["Real Returns", "Conditional Volatility"])
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def simulate_garch_returns(garch_parameters, num_simulations, num_days_to_simulate):
    """Simulate returns using GARCH parameters."""
    sim_mod = arch_model(None, p=p, q=q, dist="skewt", vol="GARCH")
    sim_df = pd.DataFrame(index=range(num_days_to_simulate), columns=range(num_simulations))
    for i in range(num_simulations):
        simulated_returns = sim_mod.simulate(garch_parameters, num_days_to_simulate)
        sim_df[sim_df.columns[i]] = simulated_returns['data']
    return sim_df

# Constants
symbol = "^FTSE"
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2022, 12, 31)
p = 3
q = 3

# Retrieve stock data
print("Retrieving data from Yahoo Finance")
ftse_data = retrieve_stock_data(symbol, start_date, end_date)

# Plot and save trends
plot_stock_data(ftse_data["Close"], "Close", os.path.join(output_folder, "ftse_close.png"))

# Calculate returns
print("Calculating returns")
returns = calculate_returns(ftse_data["Close"])

# Plot and save returns and squared returns
plot_stock_data(returns, "Returns", os.path.join(output_folder, "returns.png"))
plot_stock_data(returns**2, "Squared Returns", os.path.join(output_folder, "returns2.png"))

# Generate and save PACF plot of squared returns
print("Generating PACF plot")
plot_pacf_squared_returns(returns, os.path.join(output_folder, "pacf.png"))

# Fit GARCH model
print("Fitting GARCH model")
garch_result = fit_garch_model(returns, p, q)

# Plot and save GARCH model results
plot_garch_results(garch_result, os.path.join(output_folder, "garch_results.png"))

# Plot real returns vs. conditional volatility from GARCH result
plot_real_vs_predicted_returns(returns, garch_result.conditional_volatility, os.path.join(output_folder, "garch_plot.png"))

# Generate and save simulated returns from GARCH model
print("Simulating GARCH returns")
simulated_returns = simulate_garch_returns(garch_result.params, num_simulations=100, num_days_to_simulate=1000)
simulated_returns.plot(legend=False, figsize=(15, 5))
plt.title("GARCH Simulation")
plt.savefig(os.path.join(output_folder, "simu_100.png"), bbox_inches='tight')
plt.close()

print("Done with GARCH analysis!")
