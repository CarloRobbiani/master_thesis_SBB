import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("simulator/normal_weather.csv")

print(df["line"].unique())


for name, group in df.groupby('line'):
    plt.plot(group.index, group['simulated_delay'], label=name)

plt.legend()
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Values per line")
plt.show()