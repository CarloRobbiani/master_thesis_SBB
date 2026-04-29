import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("simulator/normal_weather.csv")

#print(df["COMMERCIAL_LINE_NUMBER_DESIGNATION"].unique())


for name, group in df.groupby('COMMERCIAL_LINE_NUMBER_DESIGNATION'):
    plt.plot(group.index, group['SIMULATED_DELAY'], label=name)

plt.legend()
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Values per line")
plt.show()