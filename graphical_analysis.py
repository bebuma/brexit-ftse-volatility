# %%
import os
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

absolute_path = os.path.abspath(__file__)
raw_data_folder_path = os.path.dirname(absolute_path) + "/raw_data/"
output1_folder_path = os.path.dirname(absolute_path) + "/output_part_1/"
output2_folder_path = os.path.dirname(absolute_path) + "/output_part_2/"

# %%
def get_raw_data(file_name=str, file_path=raw_data_folder_path):
    """get raw dataframe by put file under raw_data folder"""
    try:
        file_path = raw_data_folder_path + file_name
        if file_name.split(".")[1] == "csv":
            df = pd.read_csv(file_path)
        elif file_name.split(".")[1] == "xlsx":
            df = pd.read_excel(file_path)
        else:
            print("Please change raw data file to .csv or .xlsx")
        # converse date/month column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Month" in df.columns:
            df["Date"] = pd.to_datetime(
                df['Month'], format='%Y-%m').add(pd.offsets.MonthEnd(0))
            del df["Month"]
        else:
            print("Please change raw data file to .csv or .xlsx")
        return df
    except ImportError:
        raise ImportError("Cannot get raw data from filename given")


def get_data(file_name=str, file_path=raw_data_folder_path):
    """get raw data and clean raw data"""
    df = get_raw_data(file_name, file_path)
    # sort old to new
    df = df.sort_values(by='Date').reset_index(drop=True)
    print(f"Data file: {file_name}\n{df}\n")
    return df


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
    """generate volatility (return)"""
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


def gen_graph(line_input, df_announce, name=str, col_name=str, y="all"):
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
    plt.savefig(output1_folder_path + name + "_" +
                col_name.lower() + "_" + str(y), bbox_inches='tight')
    plt.clf()
    plt.close()


def gen_multi_graph(df, df_vol, df_announce, line_name, y):
    """generate multiple graph with annoucement mark date"""
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    axs[0].plot(df['Date'], df['Close'])
    axs[1].plot(df_vol['Date'], df_vol['vol'])
    for xc in df_announce["Date"]:
        axs[0].axvline(xc, color="red", ls="dotted")
        axs[1].axvline(xc, color="red", ls="dotted")
    fig.set_size_inches(15, 6)
    fig.savefig(output1_folder_path + line_name +
                "_" + str(y), bbox_inches='tight')
    fig.clear(True)


# %%
start = dt.datetime(2016,1,1)
end = dt.datetime(2022,12,31)
# equity market
# ftse = get_data("FTSE100.csv")
# s_and_p = get_data("S&P500.csv")
# eurostoxx = get_data("EuroStoxx50.csv")
ftse = yf.download("^FTSE", start, end).reset_index()
# ftse["Date"] = ftse["Date"].dt.date
s_and_p = yf.download("^GSPC", start, end).reset_index()
# s_and_p["Date"] = s_and_p["Date"].dt.date
eurostoxx = yf.download("^STOXX50E", start, end).reset_index()
# eurostoxx["Date"] = eurostoxx["Date"].dt.date
ftse_vol = gen_vol(ftse)
s_and_p_vol = gen_vol(s_and_p)
eurostoxx_vol = gen_vol(eurostoxx)
# rate
libor_on = get_multi_data("LIBORUSDON.csv")
libor_1m = get_multi_data("LIBORUSD1M.csv")
libor_3m = get_multi_data("LIBORUSD3M.csv")
libor_6m = get_multi_data("LIBORUSD6M.csv")
libor_12m = get_multi_data("LIBORUSD12M.csv")
reer = get_data("REER.xlsx")
# google search
google = get_data("google-search.csv")
# announcement
announce = get_data("announcement.xlsx")
# announce["Date"] = announce["Date"].dt.date

# %%
# Graphical Analysis
for y in list(range(2016, 2022)):
    # equity market
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
        filter_announce = gen_filter_df(announce, y)

        # # plot individual graph
        # gen_graph(filter_df, filter_announce, line_name,  "Close", y)
        # gen_graph(filter_df_vol, filter_announce, line_name, "vol", y)

        # plot sub graph
        gen_multi_graph(filter_df, filter_df_vol, filter_announce, line_name, y)
    print(f"Finished generating equity market in {y}")

    # rate
    filter_annouce = gen_filter_df(announce, y)
    filter_libor_on = gen_filter_df(libor_on, y)
    filter_libor_1m = gen_filter_df(libor_1m, y)
    filter_libor_3m = gen_filter_df(libor_3m, y)
    filter_libor_6m = gen_filter_df(libor_6m, y)
    filter_libor_12m = gen_filter_df(libor_12m, y)
    libor_dict = {"ON": [filter_libor_on, "orange"],
                  "1M": [filter_libor_1m, "gold"],
                  "3M": [filter_libor_3m, "darkkhaki"],
                  "6M": [filter_libor_6m, "yellowgreen"],
                  "12M": [filter_libor_12m, "darkseagreen"]}
    gen_graph(libor_dict, filter_annouce, "libor", "Close", y)
    print(f"Finished generating libor rate in {y}")

    # reer
    filter_annouce = gen_filter_df(announce, y)
    filter_reer = gen_filter_df(reer, y)
    gen_graph(filter_reer, filter_annouce, "reer", "Close", y)
    print(f"Finished generating reer in {y}")

    # Google search
    filter_annouce = gen_filter_df(announce, y)
    filter_google = gen_filter_df(google, y)
    gen_graph(filter_google, filter_annouce, "google", "Value", y)
    print(f"Finished generating google search in {y}\n")

print("\nFinished generating all graphs!")
