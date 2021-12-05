import pandas as pd

df = pd.read_csv("data_split.csv")
print(df.groupby("Set").count()/df.count()[1])
