import pandas as pd

#Load Footballers stats from .csv file
stats = pd.read_csv('PlayersStats.csv', encoding='Windows-1250', index_col=1, sep=';')

# Sprawdzenie ogólnych informacji o tabeli
print(stats.info())
print(stats.head())

# Data Preprocessing
rezerwowi = stats["Min"] < 500
stats = stats.loc[~rezerwowi]
stats = stats.drop(["Squad", "Comp", "Born", "Rk", "Nation", "Age", "MP", "Starts", "Min", "90s"], axis=1)
stats = stats.dropna()
stats["Pos"] = stats["Pos"].str.slice(stop=2)
print(stats.head())
