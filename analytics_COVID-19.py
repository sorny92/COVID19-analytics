import pandas as pd
import matplotlib.pyplot as plt



confirmed = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
deaths = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv")
recovered = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv")

print("List of available countries")
print(pd.unique(confirmed["Country/Region"]))

confirmed.info()

interesting_countries = ["Mainland China", "US", "Italy", "UK", "Spain", "Netherlands"]
interesting_countries2 = ["Spain"]
interesting_rows = confirmed["Country/Region"].isin(interesting_countries2)

data = confirmed[interesting_rows].iloc[:, 4:]
dates = data.columns.values
print(data.values[0])

plot = plt.plot(dates, data.values[0])
plt.yscale("log")
plt.grid()
plt.waitforbuttonpress()