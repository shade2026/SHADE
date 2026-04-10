import pandas as pd
import numpy as np

lon_size = {
    "Humidity": [115.05, 115.65],
    "TrafficFlow": [38.39, 38.80],
    'NO2': [115.972, 117.12],
    "Temperature": [115.05, 115.65]
}

lat_size = {
    "Humidity": [39.05, 40.45],
    "TrafficFlow": [-121.54, -121.15],
    'NO2': [39.58, 40.499],
    "Temperature": [39.05, 40.45]
}


def pro_lon(lon, size):
    return (lon - size[0]) / (size[1] - size[0]) - 0.5


def pro_lat(lat, size):
    return (lat - size[0]) / (size[1] - size[0]) - 0.5


def space_features(datas, dataset):
    data = []
    for [lon, lat] in datas:
        data.append([pro_lon(lon, lon_size[dataset]), pro_lat(lat, lat_size[dataset])])
    return data
