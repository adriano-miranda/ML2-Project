from river import feature_extraction, linear_model, metrics, preprocessing, stats, utils
import requests, pandas
from pandas import json_normalize

date = 1708452000 #Thu, 20 Feb 2024 18:0:0 GMT
cnt = 169
url = "https://history.openweathermap.org/data/2.5/history/city?lat=43.36&lon=-8.41&start={}&cnt={}&units=metric&appid=ab105e6ff8fb6bb9aef4b07d7e060f98"

size = 0
data = []

while size < 8000:
    size += cnt

    with requests.get(url.format(date, cnt), stream=True) as response:
        jsonData = response.json()['list']

        if len(jsonData) != cnt:
            print("Reached end of API feed")
            break

        def f(e):
            e['weather'] = e['weather'][0]
            return e

        data += [f(e) for e in jsonData]

    date += cnt*3600

data = json_normalize(data)

print(data.describe())

data.to_csv("data.csv")
