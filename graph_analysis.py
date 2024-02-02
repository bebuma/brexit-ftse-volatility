import os
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

absolute_path = os.path.abspath(__file__)
raw_data_folder_path = os.path.dirname(absolute_path) + "/raw_data/"
output_folder_path = os.path.dirname(absolute_path) + "/output_graph/"

def get_raw_data(file_name=str, file_path=raw_data_folder_path):
    """get raw dataframe by puting file name that under raw_data folder"""
    try:
        file_path = raw_data_folder_path + file_name
        # validate file format
        if file_name.split(".")[1] == "csv":
            df = pd.read_csv(file_path)
        elif file_name.split(".")[1] == "xlsx":
            df = pd.read_excel(file_path)
        else:
            print("Please change raw data file to .csv or .xlsx")
        # validate date value in date column
        # converse date/month column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Month" in df.columns:
            df["Date"] = pd.to_datetime(
                df['Month'], format='%Y-%m').add(pd.offsets.MonthEnd(0))
            del df["Month"]
        else:
            print("Please change raw data file to .csv or .xlsx")
        # sort old to new
        df = df.sort_values(by='Date').reset_index(drop=True)
        print(f"Data file: {file_name}\n{df}\n")
        return df
    except ImportError:
        raise ImportError("Cannot get raw data from filename given")


def get_multi_data(file_name=str, file_path=raw_data_folder_path):
    """get multi files and clean raw data"""
    df_all = []
    for y in list(range(2016, 2022)):
        new_file_name = file_name.split(
            ".")[0] + "-" + str(y) + "." + file_name.split(".")[1]
        df = get_raw_data(new_file_name, file_path)
        df_all.append(df)
    final_df = pd.concat(df_all)
    final_df = final_df.sort_values(by='Date').reset_index(drop=True)
    final_df["Close"] = final_df["Close"].str.split(
        "%", n=1, expand=True)[0].astype(float)
    print(f"Data file: {file_name}\n{final_df}\n")
    return final_df


def gen_vol(df):
    """generate returns (volatility)"""
    df["vol"] = ((df["Close"] / df["Close"].shift(1)) -
                 1).dropna(how='all') * 100
    return df


def gen_filter_df(df, y_start=int, y_end=""):
    """filter dataframe with given year"""
    if not y_end:
        y_end = y_start
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    filter_df = df.loc[(df['Date'] >= f'{y_start}-01-01')
                     & (df['Date'] <= f'{y_end}-12-31')]
    return filter_df


def gen_graph(line_input, df_announce, line_name=str, col_name=str, y="all"):
    """generate individual graph with annoucement mark date"""
    plt.figure(figsize=(15, 5))
    if type(line_input) == dict:
        for item in line_input:
            plt.plot(line_input[item][0]["Date"], line_input[item][0][col_name], label=item, color=line_input[item][1])
        plt.legend()
    else:
        plt.plot(line_input["Date"], line_input[col_name])
    for xc in df_announce["Date"]:
        plt.axvline(x=xc, color="red", ls="dotted")
    plt.title("Google search 'BREXIT'")
    plt.savefig(output_folder_path + line_name + "_" +
                col_name.lower() + "_" + str(y), bbox_inches='tight')
    plt.clf()
    plt.close()


def gen_multi_graph_multi_line(line_input_1, line_input_2, df_announce, line_name=str, col_name=str, y="all"):
    """generate multiple line on 2 graphs on 1 figure with annoucement mark date"""
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    for item in line_input_1:
        axs[0].plot(line_input_1[item][0]["Date"], line_input_1[item][0]["Close"], label=item, color=line_input_1[item][1])

    for item in line_input_2:
        axs[1].plot(line_input_2[item][0]["Date"], line_input_2[item][0][col_name], label=item, color=line_input_2[item][1])

    axs[1].legend()

    for xc in df_announce["Date"]:
        axs[0].axvline(xc, color="red", ls="dotted")
        axs[1].axvline(xc, color="red", ls="dotted")
    fig.set_size_inches(15, 5)
    fig.suptitle(f"LIBOR during {y}")
    axs[0].set_title("Rates", y=1.0, pad=-14, position=(0.05,0.6))
    axs[1].set_title("Changes", y=1.0, pad=-14, position=(0.05,0.6))
    if y == 2017:
        axs[1].set_title("Changes", y=1.0, pad=-14, position=(0.12,0.6))
    fig.savefig(output_folder_path + "libor_vol" +
                "_" + str(y), bbox_inches='tight')
    fig.clear(True)
    plt.close()


def gen_multi_graph(df, df_vol, df_announce, line_name, y):
    """generate 1 line 2 graphs on 1 figure with annoucement mark date"""
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    axs[0].plot(df['Date'], df['Close'])
    axs[1].plot(df_vol['Date'], df_vol['vol'])
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
        axs[0].set_title('Index', y=1.0, pad=-14, position=(0.05,0.6))
        axs[1].set_title('Changes', y=1.0, pad=-14, position=(0.05,0.6))
    else:
        if line_name == "ftse":
            _name = "FTSE 100"
        if line_name == "s&p":
            _name = "S&P 500"
        if line_name == "euro":
            _name = "EuroStoxx 50"
        fig.suptitle(f"{_name} Index during {y}")
        axs[0].set_title('Index', y=1.0, pad=-14, position=(0.05,0.6))
        axs[1].set_title('Returns', y=1.0, pad=-14, position=(0.05,0.6))
    fig.savefig(output_folder_path + line_name +
                "_" + str(y), bbox_inches='tight')
    fig.clear(True)
    plt.close()
    del fig


# data for equity market

# define period for equity market
start = dt.datetime(2016, 1, 1)
end = dt.datetime(2022, 12, 31)
# get data from yahoo finance
ftse = yf.download("^FTSE", start, end).reset_index()
s_and_p = yf.download("^GSPC", start, end).reset_index()
eurostoxx = yf.download("^STOXX50E", start, end).reset_index()

# data for libor rate

# get libor rate from raw data file
libor_on = get_multi_data("LIBORUSDON.csv")
libor_1m = get_multi_data("LIBORUSD1M.csv")
libor_3m = get_multi_data("LIBORUSD3M.csv")
libor_6m = get_multi_data("LIBORUSD6M.csv")
libor_12m = get_multi_data("LIBORUSD12M.csv")

# data for reer

# get reer from raw data file
reer = get_raw_data("REER.xlsx")

# generate returns

# find return and define the new dataframe
ftse_vol = gen_vol(ftse)
s_and_p_vol = gen_vol(s_and_p)
eurostoxx_vol = gen_vol(eurostoxx)
libor_on_vol = gen_vol(libor_on)
libor_1m_vol = gen_vol(libor_1m)
libor_3m_vol = gen_vol(libor_3m)
libor_6m_vol = gen_vol(libor_6m)
libor_12m_vol = gen_vol(libor_12m)
reer_vol = gen_vol(reer)

# other data

# get google search
google = get_raw_data("google-search.csv")
# get announcement
announce = get_raw_data("announcement.xlsx")

# Graphical Analysis
for y in list(range(2016,2021)):
    filter_annouce = gen_filter_df(announce, y)
    # equity market filter
    for line_name in ["ftse", "s&p", "euro"]:
        if line_name == "ftse":
            df_vol = ftse_vol.copy()
            filter_df = gen_filter_df(ftse, y)
        elif line_name == "s&p":
            df_vol = s_and_p_vol.copy()
            filter_df = gen_filter_df(s_and_p, y)
        elif line_name == "euro":
            df_vol = eurostoxx_vol.copy()
            filter_df = gen_filter_df(eurostoxx, y)
        filter_df_vol = gen_filter_df(df_vol, y)
    
        # plot sub graph
        gen_multi_graph(filter_df, filter_df_vol, filter_annouce, line_name, y)
    print(f"Finished generating equity market in {y}")

    # rate filter
    filter_libor_on = gen_filter_df(libor_on, y)
    filter_libor_1m = gen_filter_df(libor_1m, y)
    filter_libor_3m = gen_filter_df(libor_3m, y)
    filter_libor_6m = gen_filter_df(libor_6m, y)
    filter_libor_12m = gen_filter_df(libor_12m, y)
    # build dictionary to save individual variables
    libor_dict = {"ON": [filter_libor_on, "blue"],
                  "1M": [filter_libor_1m, "green"],
                  "3M": [filter_libor_3m, "pink"],
                  "6M": [filter_libor_6m, "yellow"],
                  "12M": [filter_libor_12m, "grey"]}
    # rate volatility filter
    filter_libor_on_vol = gen_filter_df(libor_on_vol, y)
    filter_libor_1m_vol = gen_filter_df(libor_1m_vol, y)
    filter_libor_3m_vol = gen_filter_df(libor_3m_vol, y)
    filter_libor_6m_vol = gen_filter_df(libor_6m_vol, y)
    filter_libor_12m_vol = gen_filter_df(libor_12m_vol, y)
    # build dictionary to save individual variables
    libor_dict_vol = {"ON": [filter_libor_on_vol, "blue"],
                  "1M": [filter_libor_1m_vol, "green"],
                  "3M": [filter_libor_3m_vol, "pink"],
                  "6M": [filter_libor_6m_vol, "yellow"],
                  "12M": [filter_libor_12m_vol, "grey"]}
    gen_multi_graph_multi_line(libor_dict, libor_dict_vol, filter_annouce, "libor_vol", "vol", y)
    print(f"Finished generating libor rate volatility in {y}")

# plot graph for all period

# reer
filter_annouce = gen_filter_df(announce, 2016, 2021)
filter_reer = gen_filter_df(reer, 2016, 2021)
filter_reer_vol = gen_filter_df(reer_vol, 2016, 2021)
gen_multi_graph(filter_reer, filter_reer_vol, filter_annouce, "reer_vol", "all")
print(f"Finished generating reer")

# Google search
filter_annouce = gen_filter_df(announce, 2016, 2021)
filter_google = gen_filter_df(google, 2016, 2021)
gen_graph(filter_google, filter_annouce, "google", "Value", "all")
print(f"Finished generating google search volume\n")

print("Finished generating all graphs!")