from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import numpy as np
import pandas as pd
import hashlib

from helpers import interpolate_zero_values

data_ = {
    'confirmed': pd.read_csv(
        "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"),
    'deaths': pd.read_csv(
        "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"),
    'recovered': pd.read_csv(
        "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
}


def preprocess(data_dict):
    for d in data_dict.keys():
        data_dict[d] = data_dict[d].drop(columns=['Lat', 'Long'])
    return data_dict


def group_by(datatype: pd.DataFrame, column: str):
    return datatype.groupby(column, as_index=False).sum()


# print("List of available countries")
# print(pd.unique(data['confirmed']["Country/Region"]))

# interesting_countries = ["China", "US", "Italy", "United Kingdom", "Spain", "Netherlands"]
interesting_countries = ["US", "Italy", "United Kingdom", "Spain", "Netherlands"]
# interesting_countries = ["China", "Italy", "United Kingdom", "Spain", "Netherlands"]
# interesting_countries = ["US", "United Kingdom", "Italy", "United Kingdom", "Spain", "Netherlands", "Germany"]
# interesting_countries = ["US", "China"]
interesting_countries = ["US", "China", "Italy", "United Kingdom", "Spain", "Netherlands", "Germany", "France", "Portugal"]


def plot_basic_logaritmic_data(data: pd.DataFrame, interesting_rows, aggregated: bool = False):
    data = data[interesting_rows].iloc[:, :]
    dates = data.columns.values[4:]
    print(data.values[0])

    fig = plt.figure(figsize=(20, 10))
    for c in range(len(data.index)):
        label = "{}-{}".format(data.values[c, 0], data.values[c, 1])
        if aggregated:
            values = data.values[c, 4:]
        else:
            values = np.concatenate(([0], np.diff(data.values[c, 4:])))
        if data.values[c, 1] == "US":
            plt.plot(dates, values, label=label, linestyle='dashed')
        elif data.values[c, 1] == "China":
            plt.plot(dates, values, label=label, marker="o")
        else:
            plt.plot(dates, values, label=label)
    plt.legend(prop={"size": 6})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    # plt.savefig(dt_string + ".png")


def from_day_zero(data: pd.DataFrame, interesting_rows, aggregated: bool = False):
    day_zero_n_patients = 12
    data = data[interesting_rows].iloc[:, :]

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        label = data.values[c, 0]
        color = '#{}'.format(hashlib.md5(label.encode()).hexdigest()[:6])
        print(color)
        if aggregated:
            values = data.values[c, 4:][data.iloc[c, 4:] > day_zero_n_patients]
        else:
            values = np.array(np.diff(data.values[c, 4:][data.iloc[c, 4:] > day_zero_n_patients]), dtype=np.int)
            values = interpolate_zero_values(values)
            values = np.concatenate(([0], values))

        if data.values[c, 0] in ["US","China"]:
            plt.plot(values, label=label, linestyle='dashed', color=color)
        else:
            plt.plot(values, label=label, marker='o', color=color)
    plt.legend(prop={"size": 10})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')
    plt.tight_layout()

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    # plt.savefig(dt_string + ".png")

data_ = preprocess(data_)

confirmed = group_by(data_['confirmed'], 'Country/Region')
interesting_rows = confirmed["Country/Region"].isin(interesting_countries)

deaths = group_by(data_['deaths'], 'Country/Region')

# print(confirmed)
from_day_zero(confirmed, interesting_rows)
from_day_zero(confirmed, interesting_rows, aggregated=True)
# plot_basic_logaritmic_data(confirmed, interesting_rows)
from_day_zero(deaths, interesting_rows)
from_day_zero(deaths, interesting_rows, aggregated=True)
#
# from_day_zero(recovered, interesting_rows)
# from_day_zero(recovered, interesting_rows, aggregated=True)
#
# predictive_model(confirmed, interesting_rows, day_zero_n_patients=40, days_in_future=50)
# predictive_model(deaths, interesting_rows, day_zero_n_patients=1, days_in_future=30)
plt.show()
