from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fit_a_curver(data: pd.DataFrame, interesting_rows, aggregated: bool = False):
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

def predictive_model(data: pd.DataFrame, interesting_rows, day_zero_n_patients: int = 20, days_in_future: int = 30,
                     aggregated: bool = False):
    data = data[interesting_rows].iloc[:, :]
    from lmfit.models import StepModel, ExponentialModel

    fig = plt.figure(figsize=(10, 5))
    for c in range(len(data.index)):
        if aggregated:
            values = data.values[c, 4:][data.iloc[c, 4:] > day_zero_n_patients]
        else:
            values = np.concatenate(([0], np.diff(data.values[c, 4:][data.iloc[c, 4:] > day_zero_n_patients])))

        n = values.shape[0]
        x = np.asarray(range(values.shape[0]), dtype='float64')
        y = np.asarray(values, dtype='float64')

        if len(x) == 0:
            continue

        label = "{}-{}".format(data.values[c, 0], data.values[c, 1])
        plt.plot(x, y, label=label)
        if data.values[c, 1] in ["China", "US"]:
            continue

        try:
            model_step = StepModel()
            model_exp = ExponentialModel()
            params_step = model_step.guess(y, x=x)
            params_exp = model_exp.guess(y, x=x)

            result_step = model_step.fit(y, params_step, x=x)
            result_exp = model_exp.fit(y, params_exp, x=x)
        except Exception:
            continue
        x_pred = np.asarray(range(days_in_future))
        plt.plot(x_pred, model_step.eval(result_step.params, x=x_pred), ':', label='fit-{}'.format(label))
        plt.plot(x_pred, model_exp.eval(result_exp.params, x=x_pred), '.', label='fit-{}'.format(label))
        # print(result.fit_report())
        # result.plot_fit()
    plt.legend(prop={"size": 7})
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(which='both')
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    # plt.savefig(dt_string + ".png")