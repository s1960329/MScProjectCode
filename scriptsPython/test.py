from misc import dataPath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



data = pd.read_csv("data/Raw/FullDataClassified.csv")

# plt.hist([data[data["isSignal"] == 1.0]["SignalProbGBdouble"], data[data["isSignal"] == 0.0]["SignalProbGBdouble"]], bins=100, density = False, histtype = "step")
# plt.show()

l = []
for cut in np.linspace(0,1,1000):
    eBDT  = len(data[ (data["SignalProbGBdouble"] > cut) & (data["isSignal"] == 1.0) ]) / len(data[data["isSignal"] == 1.0])
    Bcomb = len(data[ (data["SignalProbGBdouble"] > cut) ])
    l.append(6000*eBDT / np.sqrt(6000*eBDT + Bcomb))


plt.plot(l)
plt.show()

