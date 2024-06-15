import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf


class GARCHAnalysis:
    def __init__(self, symbol, start_date, end_date, p, q, output_folder):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.p = p
        self.q = q
        self.output_folder = output_folder
        self.data = None
        self.returns = None
        self.garch_result = None

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def retrieve_stock_data(self):
        """Retrieve stock data from Yahoo Finance."""
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)

    def calculate_returns(self):
        """Calculate daily returns from closing prices."""
        self.returns = 100 * self.data["Close"].pct_change().dropna()

    def plot_data(self, data, title, filename):
        """Plot and save stock data."""
        plt.figure(figsize=(15, 5))
        plt.plot(data)
        plt.title(title)
        plt.savefig(os.path.join(self.output_folder, filename), bbox_inches="tight")
        plt.close()

    def plot_pacf_squared_returns(self, filename):
        """Generate and save PACF plot of squared returns."""
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_pacf(self.returns**2, method="ywm", ax=ax)
        plt.savefig(os.path.join(self.output_folder, filename), bbox_inches="tight")
        plt.close()

    def fit_garch_model(self):
        """Fit a GARCH model to returns and return the result."""
        model = arch_model(self.returns, p=self.p, q=self.q, dist="skewt", vol="GARCH")
        self.garch_result = model.fit(disp="off")

    def plot_garch_results(self, filename):
        """Plot and save GARCH model results."""
        fig = self.garch_result.plot()
        fig.set_size_inches(15, 5)
        fig.savefig(os.path.join(self.output_folder, filename), bbox_inches="tight")
        plt.close()

    def plot_real_vs_predicted_returns(self, filename):
        """Plot and save real returns vs. conditional volatility from GARCH result."""
        plt.figure(figsize=(15, 5))
        plt.plot(self.returns, label="Real Returns")
        plt.plot(
            self.garch_result.conditional_volatility, label="Conditional Volatility"
        )
        plt.title("GARCH Model")
        plt.legend(["Real Returns", "Conditional Volatility"])
        plt.savefig(os.path.join(self.output_folder, filename), bbox_inches="tight")
        plt.close()

    def simulate_garch_returns(self, num_simulations, num_days_to_simulate):
        """Simulate returns using GARCH parameters."""
        sim_mod = arch_model(None, p=self.p, q=self.q, dist="skewt", vol="GARCH")
        sim_df = pd.DataFrame(
            index=range(num_days_to_simulate), columns=range(num_simulations)
        )
        for i in range(num_simulations):
            simulated_returns = sim_mod.simulate(
                self.garch_result.params, num_days_to_simulate
            )
            sim_df[sim_df.columns[i]] = simulated_returns["data"]
        sim_df.plot(legend=False, figsize=(15, 5))
        plt.title("GARCH Simulation")
        plt.savefig(
            os.path.join(self.output_folder, "simu_100.png"), bbox_inches="tight"
        )
        plt.close()


def main():
    # Constants
    symbol = "^FTSE"
    start_date = dt.datetime(2000, 1, 1)
    end_date = dt.datetime(2022, 12, 31)
    p = 3
    q = 3

    # Define output folder
    absolute_path = os.path.abspath(__file__)
    output_folder = os.path.dirname(absolute_path) + "/output_garch/"

    # Create GARCHAnalysis object
    garch_analysis = GARCHAnalysis(symbol, start_date, end_date, p, q, output_folder)

    # Run analysis
    print("Retrieving data from Yahoo Finance")
    garch_analysis.retrieve_stock_data()

    # Plot and save trends
    garch_analysis.plot_data(garch_analysis.data["Close"], "Close", "ftse_close.png")

    # Calculate returns
    print("Calculating returns")
    garch_analysis.calculate_returns()

    # Plot and save returns and squared returns
    garch_analysis.plot_data(garch_analysis.returns, "Returns", "returns.png")
    garch_analysis.plot_data(
        garch_analysis.returns**2, "Squared Returns", "returns2.png"
    )

    # Generate and save PACF plot of squared returns
    print("Generating PACF plot")
    garch_analysis.plot_pacf_squared_returns("pacf.png")

    # Fit GARCH model
    print("Fitting GARCH model")
    garch_analysis.fit_garch_model()

    # Plot and save GARCH model results
    garch_analysis.plot_garch_results("garch_results.png")

    # Plot real returns vs. conditional volatility from GARCH result
    garch_analysis.plot_real_vs_predicted_returns("garch_plot.png")

    # Generate and save simulated returns from GARCH model
    print("Simulating GARCH returns")
    garch_analysis.simulate_garch_returns(
        num_simulations=100, num_days_to_simulate=1000
    )

    print("Done with GARCH analysis!")


if __name__ == "__main__":
    main()
