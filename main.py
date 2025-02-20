from river import feature_extraction, linear_model, metrics, preprocessing, stats, utils
import requests, pandas
from pandas import json_normalize

date = 1708617600 #Thu, 22 Feb 2024 17:0:0 GMT
cnt = 169 # number of hourly weather reports we want per call (169 is the maximum)
url = "https://history.openweathermap.org/data/2.5/history/city?lat=43.36&lon=-8.41&start={}&cnt={}&units=metric&appid=ab105e6ff8fb6bb9aef4b07d7e060f98"

MAX_SIZE = 8000

# accumulators
size = 0
data = []

while size < MAX_SIZE:
    size += cnt

    with requests.get(url.format(date, cnt), stream=True) as response:
        # checking if the request was not correct
        if response.status_code != 200:
            print("Reached end of API feed")
            break

        # getting the hourly list of weather reports
        jsonData = response.json()['list']

        # mapping function that unpacks the weather field
        # necessary for conversion into a pandas table later
        def f(e):
            e['weather'] = e['weather'][0]
            return e

        data += [f(e) for e in jsonData]

        # new date is the most recent date retrieved by the API + an hour (3600 seconds)
        date = int(jsonData[-1]['dt']) + 3600

# convert json structured data into a table with pandas
data = json_normalize(data)

print(data.describe())

# save pandas table to disk
data.to_csv("data.csv")
