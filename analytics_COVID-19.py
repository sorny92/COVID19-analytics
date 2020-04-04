from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

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

interesting_countries = ["US", "China", "Italy", "United Kingdom", "Spain", "Netherlands", "Germany", "France",
                         "Portugal", "Austria", "Brazil", "Turkey", "Russia"]

population = {"US": 327200000, "China": 1386000000, "Italy": 60255000, "United Kingdom": 67036000, "Spain": 47195000,
              "Netherlands": 17409000, "Germany": 83241000, "France": 64869000, "Portugal": 10240000, "Austria": 8822000,
              "Brazil":209000000, "Russia": 144500000, "Turkey":80810000}


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


def from_day_zero(data: pd.DataFrame, interesting_rows, y_label: str, day_zero_n_patients: int = 20,
                  aggregated: bool = False):
    data = data[interesting_rows].iloc[:, :]

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        label = data.values[c, 0]
        color = '#{}'.format(hashlib.md5(label.encode()).hexdigest()[:6])
        values = data.values[c, 1:][data.iloc[c, 1:] >= day_zero_n_patients]
        if not aggregated:
            values = np.array(np.diff(values), dtype=np.int)
            values = interpolate_zero_values(values)
            values = np.concatenate(([0], values))


        values_smooth = gaussian_filter1d(list(values), sigma=1)
        if data.values[c, 0] in ["US", "China"]:
            plt.plot(values_smooth, label=label, linestyle='dashed', color=color)
        else:
            plt.plot(values_smooth, label=label, color=color)
    plt.legend(prop={"size": 10})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')
    plt.ylabel(y_label)
    plt.xlabel("Days since case nº{}".format(day_zero_n_patients))
    plt.tight_layout()

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    plt.savefig("out/"+ y_label + ".svg")


def from_day_zero_over_population(data: pd.DataFrame, interesting_rows, population_dict: dict, y_label: str,
                                  day_zero_n_patients: int = 20, aggregated: bool = False):
    data = data[interesting_rows].iloc[:, :]
    plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        label = data.values[c, 0]
        color = '#{}'.format(hashlib.md5(label.encode()).hexdigest()[:6])
        values = data.values[c, 1:][data.iloc[c, 1:] >= day_zero_n_patients]
        values *= 1000000
        values /= float(population_dict[label])
        if not aggregated:
            values = np.array(np.diff(values), dtype=np.float64)
            values = interpolate_zero_values(values)
            values = np.concatenate(([0], values))

        values_smooth = gaussian_filter1d(list(values), sigma=1)
        if data.values[c, 0] in ["US", "China"]:
            plt.plot(values_smooth, label=label, linestyle='dashed', color=color)
        else:
            plt.plot(values_smooth, label=label, color=color)
    plt.legend(prop={"size": 10})
    # plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')
    plt.ylabel(y_label)
    plt.xlabel("Days since case nº{}".format(day_zero_n_patients))
    plt.tight_layout()

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    plt.savefig("out/"+ y_label + ".svg")


data_ = preprocess(data_)

confirmed = group_by(data_['confirmed'], 'Country/Region')
interesting_rows = confirmed["Country/Region"].isin(interesting_countries)

deaths = group_by(data_['deaths'], 'Country/Region')

from_day_zero(confirmed, interesting_rows, "New infections")
from_day_zero(confirmed, interesting_rows, "Total infections", aggregated=True)

from_day_zero_over_population(confirmed, interesting_rows, population, "New infections over 1M population", day_zero_n_patients=20)
from_day_zero_over_population(confirmed, interesting_rows, population, "Total infections over 1M population", day_zero_n_patients=20, aggregated=True)

from_day_zero(deaths, interesting_rows, "New deaths")
from_day_zero(deaths, interesting_rows, "Total deaths", aggregated=True)

from_day_zero_over_population(deaths, interesting_rows, population, "New deaths over 1M population")
from_day_zero_over_population(deaths, interesting_rows, population, "Total deaths over 1M population", aggregated=True)

# CASES/POPULATION
# ACTIVE CASES (INFECTIONS - DEATHS - RECOVERED)
# RATIO MUERTES A PARTIR DE RANGO DE EDAD +50/+60/+75/+80
#
# from_day_zero(recovered, interesting_rows)
# from_day_zero(recovered, interesting_rows, aggregated=True)
#
# predictive_model(confirmed, interesting_rows, day_zero_n_patients=40, days_in_future=50)
# predictive_model(deaths, interesting_rows, day_zero_n_patients=1, days_in_future=30)
plt.show()
