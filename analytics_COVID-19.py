from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit

confirmed = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
deaths = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv")
recovered = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv")

# print("List of available countries")
# print(pd.unique(confirmed["Country/Region"]))

interesting_countries = ["Mainland China", "US", "Italy", "UK", "Spain", "Netherlands"]
# interesting_countries = ["US", "Italy", "UK", "Spain", "Netherlands"]
# interesting_countries = ["Italy", "UK", "Spain", "Netherlands"]


def plot_basic_logaritmic_data(data: pd.DataFrame, interesting_data: list):
    interesting_rows = confirmed["Country/Region"].isin(interesting_data)
    data = data[interesting_rows].iloc[:, :]
    dates = data.columns.values[4:]
    print(data.values[0])

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        if data.values[c, 1] is "US":
            plot = plt.plot(dates, data.values[c, 4:],
                            label="{}-{}".format(data.values[c, 0], data.values[c, 1], linestyle='dashed'))
        else:
            plot = plt.plot(dates, data.values[c, 4:], label="{}-{}".format(data.values[c, 0], data.values[c, 1]))
    plt.legend(prop={"size": 4})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    plt.savefig(dt_string + ".png")


def from_day_zero(data: pd.DataFrame, interesting_data: list):
    interesting_rows = confirmed["Country/Region"].isin(interesting_data)
    data = data[interesting_rows].iloc[:, :]

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        label = "{}-{}".format(data.values[c, 0], data.values[c, 1])
        values = data.values[c, 4:][data.iloc[c, 4:] > 3]

        # break
        plot = plt.plot(values, label=label)
    plt.legend(prop={"size": 4})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    plt.savefig(dt_string + ".png")


def fit_a_curver(data: pd.DataFrame, interesting_data: list):
    interesting_rows = confirmed["Country/Region"].isin(interesting_data)
    data = data[interesting_rows].iloc[:, :]

    def gaus(x, a, x0, sigma):
        return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        n = data.values[c, 4:].shape[0]
        x = ar(range(data.values[c, 4:].shape[0]))
        y = data.iloc[c, 4:].diff()
        # print(y.diff())
        mean = sum(x * y) / n  # note this correction
        sigma = sum(y * (x - mean) ** 2) / n  # note this correction
        try:
            popt, pcov = curve_fit(gaus, x, data.values[c, 4:], method='dogbox')
        except RuntimeError:
            print("PETOOOOOO")
        print(popt)
        label = "{}-{}".format(data.values[c, 0], data.values[c, 1])
        values = data.values[c, 4:]
        plot = plt.plot(x, y, label=label)
        # plt.plot(x, y, 'b+:', label='data')
        plt.plot(x, gaus(x, *popt), 'ro:', label='fit')
    plt.legend(prop={"size": 4})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')


def other_fitter(data: pd.DataFrame, interesting_data: list):
    interesting_rows = confirmed["Country/Region"].isin(interesting_data)
    data = data[interesting_rows].iloc[:, :]
    from lmfit.models import LorentzianModel

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        n = data.values[c, 4:].shape[0]
        x = ar(range(data.values[c, 4:].shape[0]))
        y = data.iloc[c, 4:].diff().fillna(0)

        print(y)
        print(x)
        model = LorentzianModel()
        params = model.guess(y.values, x=x)

        result = model.fit(y, params, x=x)
        label = "{}-{}".format(data.values[c, 0], data.values[c, 1])
        plt.plot(x, y, label=label)
        x_pred = ar(range(100))
        plt.plot(x_pred, model.eval(result.params, x=x_pred), 'ro:', label='fit')
        # print(result.fit_report())
        # result.plot_fit()
    plt.legend(prop={"size": 4})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    plt.savefig(dt_string + ".png")

from_day_zero(confirmed, interesting_countries)
plt.pause(1)
plot_basic_logaritmic_data(confirmed, interesting_countries)
plt.pause(1)
from_day_zero(deaths, interesting_countries)
plt.pause(1)
plot_basic_logaritmic_data(deaths, interesting_countries)
plt.pause(1)
from_day_zero(recovered, interesting_countries)
plt.pause(1)
plot_basic_logaritmic_data(recovered, interesting_countries)
plt.pause(1)
#
# fit_a_curver(confirmed, ["Mainland China"])
# fit_a_curver(confirmed, ["Spain"])
# fit_a_curver(confirmed, ["Italy", "UK", "Spain", "Netherlands"])
# other_fitter(confirmed, interesting_countries)
plt.show()
