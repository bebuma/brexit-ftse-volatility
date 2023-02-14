import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from datetime import timedelta
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm

absolute_path = os.path.abspath(__file__)
output2_folder_path = os.path.dirname(absolute_path) + "/output_part_2/"
p=3
q=3

def run_garch_model(returns, p=p, q=q):
    """build GARCH from ARCH model and fit model to get model parameters"""
    # using GARCH p, q
    a_model = arch_model(returns, p=p, q=q, dist='skewt', vol="GARCH")
    # fit model
    result = a_model.fit(disp='off')
    return result


def output_sim_returns(garch_parameters, num_days_to_simulate=1000):
    """generate returns from given GARCH parameters"""
    sim_mod = arch_model(None, p=p, q=q, dist="skewt", vol="GARCH")
    sim_data = sim_mod.simulate(garch_parameters, num_days_to_simulate)
    return sim_data['data']


def simulate_returns(returns, num_sims=10, num_days_to_simulate=1000):
    """simulate returns from providing data"""
    sim_df = pd.DataFrame(index=range(num_days_to_simulate), columns=range(num_sims))
    result = run_garch_model(returns)
    garch_parameters = result.params
    for i in range(num_sims):
        simulated_returns = output_sim_returns(garch_parameters, num_days_to_simulate)
        sim_df[sim_df.columns[i]] = simulated_returns
    return sim_df


# get data from yahoo finance
start = dt.datetime(2000,1,1)
end = dt.datetime(2022,12,31)
print("Start retrieving data from Yahoo Finance")
ftse = yf.download("^FTSE", start, end)
# calculate returns
print("Generate returns")
returns = 100 * ftse.Close.pct_change().dropna()
# plot returns
print("Plot returns")
plt.figure(figsize=(15, 5))
plt.plot(returns)
plt.title("Returns")
plt.savefig(output2_folder_path + "returns", bbox_inches='tight')
plt.clf()
plt.close()
# plot squared returns
print("Plot squared returns")
plt.figure(figsize=(15, 5))
plt.plot(returns**2)
plt.title("Squared Returns")
plt.savefig(output2_folder_path + "returns2", bbox_inches='tight')
plt.clf()
plt.close()

# find the relationship between lag days
# partial autoregression with squared return
print("Start generate PACF")
fig, ax = plt.subplots(figsize=(15, 5))
plot_pacf(returns**2, method="ywm", ax=ax)
plt.savefig(output2_folder_path + "pacf", bbox_inches='tight')
plt.clf()
plt.close()

# GARCH model
# run model 
garch_result = run_garch_model(returns, p=p, q=q)
# fitted results
print(f"Maximum likihood method \n{garch_result.params}\n")
print(f"Summary \n{garch_result.summary()}\n")
# plot
# garch_output.plot()
fig = garch_result.plot()
print("Plot GARCH output")
fig.set_size_inches(15, 5)
fig.savefig(output2_folder_path + "garch_results", bbox_inches='tight')
fig.clear(True)
plt.close()

# model performance with real data
print("Plot real returns vs conditional volatility from GARCH result")
retuens_df = pd.DataFrame(returns)
predicted_df = pd.DataFrame(garch_result.conditional_volatility)
garch_plot = pd.concat([retuens_df, predicted_df], axis=1)
plt.figure(figsize=(15, 5))
plt.plot(garch_plot)
plt.title("GARCH Model")
plt.legend(["Real Returns", "Conditional Volatility"])
plt.savefig(output2_folder_path + "garch_plot", bbox_inches='tight')
plt.clf()
plt.close()

# generate simulation of GARCH model result
sim_df = simulate_returns(returns, num_sims=100, num_days_to_simulate=1000)
print(f"Predicted returns:\n{sim_df}\n")
sim_df.plot(legend=False, figsize=(15, 5))
plt.title("GARCH Simulation")
plt.savefig(output2_folder_path + "simu_100", bbox_inches='tight')
plt.clf()
plt.close()

print("Done volatility analysis by GARCH model!")
