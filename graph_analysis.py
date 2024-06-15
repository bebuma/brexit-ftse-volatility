import os
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, raw_data_folder_path, output_folder_path):
        self.raw_data_folder_path = raw_data_folder_path
        self.output_folder_path = output_folder_path

    def get_raw_data(self, file_name):
        """get raw dataframe by putting file name that under raw_data folder"""
        try:
            file_path = self.raw_data_folder_path + file_name
            if file_name.split(".")[1] == "csv":
                df = pd.read_csv(file_path)
            elif file_name.split(".")[1] == "xlsx":
                df = pd.read_excel(file_path)
            else:
                print("Please change raw data file to .csv or .xlsx")
                return None
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            elif "Month" in df.columns:
                df["Date"] = pd.to_datetime(df["Month"], format="%Y-%m").add(
                    pd.offsets.MonthEnd(0)
                )
                del df["Month"]
            else:
                print("Please change raw data file to .csv or .xlsx")
                return None
            df = df.sort_values(by="Date").reset_index(drop=True)
            return df
        except ImportError:
            raise ImportError("Cannot get raw data from filename given")

    def get_multi_data(self, file_name):
        """get multi files and clean raw data"""
        df_all = []
        for y in range(2016, 2022):
            new_file_name = (
                file_name.split(".")[0] + "-" + str(y) + "." + file_name.split(".")[1]
            )
            df = self.get_raw_data(new_file_name)
            df_all.append(df)
        final_df = pd.concat(df_all)
        final_df = final_df.sort_values(by="Date").reset_index(drop=True)
        final_df["Close"] = (
            final_df["Close"].str.split("%", n=1, expand=True)[0].astype(float)
        )
        return final_df

    def generate_volatility(self, df):
        """generate returns (volatility)"""
        df["vol"] = ((df["Close"] / df["Close"].shift(1)) - 1).dropna(how="all") * 100
        return df

    def filter_dataframe(self, df, y_start, y_end=""):
        """filter dataframe with given year"""
        if not y_end:
            y_end = y_start
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        filter_df = df.loc[
            (df["Date"] >= f"{y_start}-01-01") & (df["Date"] <= f"{y_end}-12-31")
        ]
        return filter_df


class GraphGenerator:
    def __init__(self, output_folder_path):
        self.output_folder_path = output_folder_path

    def generate_graph(self, line_input, df_announce, line_name, col_name, y="all"):
        """generate individual graph with announcement mark date"""
        plt.figure(figsize=(15, 5))
        if isinstance(line_input, dict):
            for item in line_input:
                plt.plot(
                    line_input[item][0]["Date"],
                    line_input[item][0][col_name],
                    label=item,
                    color=line_input[item][1],
                )
            plt.legend()
        else:
            plt.plot(line_input["Date"], line_input[col_name])
        for xc in df_announce["Date"]:
            plt.axvline(x=xc, color="red", ls="dotted")
        plt.title("Google search 'BREXIT'")
        plt.savefig(
            self.output_folder_path + line_name + "_" + col_name.lower() + "_" + str(y),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    def generate_multi_graph_multi_line(
        self, line_input_1, line_input_2, df_announce, line_name, col_name, y="all"
    ):
        """generate multiple line on 2 graphs on 1 figure with announcement mark date"""
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0)

        for item in line_input_1:
            axs[0].plot(
                line_input_1[item][0]["Date"],
                line_input_1[item][0]["Close"],
                label=item,
                color=line_input_1[item][1],
            )

        for item in line_input_2:
            axs[1].plot(
                line_input_2[item][0]["Date"],
                line_input_2[item][0][col_name],
                label=item,
                color=line_input_2[item][1],
            )

        axs[1].legend()

        for xc in df_announce["Date"]:
            axs[0].axvline(xc, color="red", ls="dotted")
            axs[1].axvline(xc, color="red", ls="dotted")
        fig.set_size_inches(15, 5)
        fig.suptitle(f"LIBOR during {y}")
        axs[0].set_title("Rates", y=1.0, pad=-14, position=(0.05, 0.6))
        axs[1].set_title("Changes", y=1.0, pad=-14, position=(0.05, 0.6))
        if y == 2017:
            axs[1].set_title("Changes", y=1.0, pad=-14, position=(0.12, 0.6))
        fig.savefig(
            self.output_folder_path + "libor_vol" + "_" + str(y), bbox_inches="tight"
        )
        fig.clear(True)
        plt.close()

    def generate_multi_graph(self, df, df_vol, df_announce, line_name, y):
        """generate 1 line 2 graphs on 1 figure with announcement mark date"""
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        axs[0].plot(df["Date"], df["Close"])
        axs[1].plot(df_vol["Date"], df_vol["vol"])
        for xc in df_announce["Date"]:
            axs[0].axvline(xc, color="red", ls="dotted")
            axs[1].axvline(xc, color="red", ls="dotted")
        if line_name in ["reer_vol"]:
            axs[0].axhline(100, color="lightgrey", zorder=1)
        if line_name in ["ftse", "s&p", "euro", "reer_vol"]:
            axs[1].axhline(0, color="lightgrey", zorder=1)
        fig.set_size_inches(15, 5)
        if line_name == "reer_vol":
            fig.suptitle("GBP REER from 2016 to 2021")
            axs[0].set_title("Index", y=1.0, pad=-14, position=(0.05, 0.6))
            axs[1].set_title("Changes", y=1.0, pad=-14, position=(0.05, 0.6))
        else:
            if line_name == "ftse":
                _name = "FTSE 100"
            if line_name == "s&p":
                _name = "S&P 500"
            if line_name == "euro":
                _name = "EuroStoxx 50"
            fig.suptitle(f"{_name} Index during {y}")
            axs[0].set_title("Index", y=1.0, pad=-14, position=(0.05, 0.6))
            axs[1].set_title("Returns", y=1.0, pad=-14, position=(0.05, 0.6))
        fig.savefig(
            self.output_folder_path + line_name + "_" + str(y), bbox_inches="tight"
        )
        fig.clear(True)
        plt.close()
        del fig


def main():
    absolute_path = os.path.abspath(__file__)
    raw_data_folder_path = os.path.dirname(absolute_path) + "/raw_data/"
    output_folder_path = os.path.dirname(absolute_path) + "/output_graph/"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    data_processor = DataProcessor(raw_data_folder_path, output_folder_path)
    graph_generator = GraphGenerator(output_folder_path)

    # Data for equity market
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2022, 12, 31)
    ftse = yf.download("^FTSE", start, end).reset_index()
    s_and_p = yf.download("^GSPC", start, end).reset_index()
    eurostoxx = yf.download("^STOXX50E", start, end).reset_index()

    # Data for LIBOR rate
    libor_on = data_processor.get_multi_data("LIBORUSDON.csv")
    libor_1m = data_processor.get_multi_data("LIBORUSD1M.csv")
    libor_3m = data_processor.get_multi_data("LIBORUSD3M.csv")
    libor_6m = data_processor.get_multi_data("LIBORUSD6M.csv")
    libor_12m = data_processor.get_multi_data("LIBORUSD12M.csv")

    # Data for REER
    reer = data_processor.get_raw_data("REER.xlsx")

    # Generate returns
    ftse_vol = data_processor.generate_volatility(ftse)
    s_and_p_vol = data_processor.generate_volatility(s_and_p)
    eurostoxx_vol = data_processor.generate_volatility(eurostoxx)
    libor_on_vol = data_processor.generate_volatility(libor_on)
    libor_1m_vol = data_processor.generate_volatility(libor_1m)
    libor_3m_vol = data_processor.generate_volatility(libor_3m)
    libor_6m_vol = data_processor.generate_volatility(libor_6m)
    libor_12m_vol = data_processor.generate_volatility(libor_12m)
    reer_vol = data_processor.generate_volatility(reer)

    # Other data
    google = data_processor.get_raw_data("google-search.csv")
    announce = data_processor.get_raw_data("announcement.xlsx")

    # Graphical Analysis
    for y in range(2016, 2021):
        filter_annouce = data_processor.filter_dataframe(announce, y)
        # Equity market filter
        for line_name in ["ftse", "s&p", "euro"]:
            if line_name == "ftse":
                df_vol = ftse_vol.copy()
                filter_df = data_processor.filter_dataframe(ftse, y)
            elif line_name == "s&p":
                df_vol = s_and_p_vol.copy()
                filter_df = data_processor.filter_dataframe(s_and_p, y)
            elif line_name == "euro":
                df_vol = eurostoxx_vol.copy()
                filter_df = data_processor.filter_dataframe(eurostoxx, y)
            filter_df_vol = data_processor.filter_dataframe(df_vol, y)
            graph_generator.generate_multi_graph(
                filter_df, filter_df_vol, filter_annouce, line_name, y
            )
        print(f"Finished generating equity market in {y}")

        # Rate filter
        filter_libor_on = data_processor.filter_dataframe(libor_on, y)
        filter_libor_1m = data_processor.filter_dataframe(libor_1m, y)
        filter_libor_3m = data_processor.filter_dataframe(libor_3m, y)
        filter_libor_6m = data_processor.filter_dataframe(libor_6m, y)
        filter_libor_12m = data_processor.filter_dataframe(libor_12m, y)
        libor_dict = {
            "ON": [filter_libor_on, "blue"],
            "1M": [filter_libor_1m, "green"],
            "3M": [filter_libor_3m, "pink"],
            "6M": [filter_libor_6m, "yellow"],
            "12M": [filter_libor_12m, "grey"],
        }
        filter_libor_on_vol = data_processor.filter_dataframe(libor_on_vol, y)
        filter_libor_1m_vol = data_processor.filter_dataframe(libor_1m_vol, y)
        filter_libor_3m_vol = data_processor.filter_dataframe(libor_3m_vol, y)
        filter_libor_6m_vol = data_processor.filter_dataframe(libor_6m_vol, y)
        filter_libor_12m_vol = data_processor.filter_dataframe(libor_12m_vol, y)
        libor_dict_vol = {
            "ON": [filter_libor_on_vol, "blue"],
            "1M": [filter_libor_1m_vol, "green"],
            "3M": [filter_libor_3m_vol, "pink"],
            "6M": [filter_libor_6m_vol, "yellow"],
            "12M": [filter_libor_12m_vol, "grey"],
        }
        graph_generator.generate_multi_graph_multi_line(
            libor_dict, libor_dict_vol, filter_annouce, "libor_vol", "vol", y
        )
        print(f"Finished generating libor rate volatility in {y}")

    # Plot graph for all period
    filter_annouce = data_processor.filter_dataframe(announce, 2016, 2021)
    filter_reer = data_processor.filter_dataframe(reer, 2016, 2021)
    filter_reer_vol = data_processor.filter_dataframe(reer_vol, 2016, 2021)
    graph_generator.generate_multi_graph(
        filter_reer, filter_reer_vol, filter_annouce, "reer_vol", "all"
    )
    print(f"Finished generating REER")

    filter_google = data_processor.filter_dataframe(google, 2016, 2021)
    graph_generator.generate_graph(
        filter_google, filter_annouce, "google", "Value", "all"
    )
    print(f"Finished generating google search volume\n")

    print("Finished generating all graphs!")


if __name__ == "__main__":
    main()
