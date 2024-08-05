import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def BinaryGini(sigProb):
    return 1 - sigProb**2 - (1-sigProb)**2

def BinaryEntropy(sigProb):
    return (-1)*sigProb*np.log2(sigProb) + (-1)*(1-sigProb)*np.log2(1-sigProb)

def plotBinaryGini():

    bins    = np.linspace(0,1,1001)
    gini    = np.array([BinaryGini(sigProb)    for sigProb in bins])
    entropy = np.array([BinaryEntropy(sigProb) for sigProb in bins])

    plt.grid(linestyle="--", alpha=0.3)
    plt.plot(bins, entropy, color="#006cd4",  label="Entropy")
    plt.plot(bins, gini,    color="#d80645",  label="Gini")
    plt.xlim(0,1)
    plt.ylim(0,1.2)

    plt.title("Purity Measures")
    plt.xlabel("Proportion of Signal on a Leaf")
    plt.ylabel("Impurity")
    plt.legend()
    plt.savefig("PurityPlot.png", dpi=227)
    plt.close()

def Importance(x):
    if x == 0: x = 0.001
    if x == 1: x = 0.999
    return 0.5*np.log( (1-x) / (x))

def plotImportance():

    bins = np.linspace(0,1,1001)
    say  = np.array([Importance(sigProb) for sigProb in bins])

    plt.plot(bins, say, color="#006cd4",  label="Importance")
    plt.grid(linestyle="--", alpha=0.3)
    plt.xlim(0,1)
    plt.ylim(-4,4)

    plt.title("AdaBoost Importance of Stump")
    plt.xlabel("Weighted Error")
    plt.ylabel("Importance")
    plt.savefig("ImportancePlot.png", dpi=227)
    plt.close()
    
def plotReweighting(initialWeight = 0.1):
    
    bins = np.array([Importance(sigProb) for sigProb in np.linspace(0,1,1001)]) 
    pos  = np.array([initialWeight*np.exp( imp) for imp in bins])
    neg  = np.array([initialWeight*np.exp(-imp) for imp in bins])

    plt.plot(bins, pos, color="#006cd4",  label="Miss")
    plt.plot(bins, neg, color="#d80645",  label="Hit")
    plt.grid(linestyle="--", alpha=0.3)
    plt.xlim(-3,3)
    plt.ylim(0,2)

    plt.title("AdaBoost New Weight of Entry")
    plt.xlabel("Importance")
    plt.ylabel("New Weight")
    plt.legend()
    plt.savefig("AdaNewWeightPlot.png", dpi=227)
    plt.close()

if __name__ == "__main__":
    plotReweighting()
