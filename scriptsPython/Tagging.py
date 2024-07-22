
import joblib
import pandas                as     pd
import numpy                 as     np
import matplotlib.pyplot     as     plt

from misc                    import fileNamesCSV, imagePath, histStyle, colors, sharedVariables, uniqueVariables
from misc                    import formatForTagging, normaliseData, roundUp


def createBackgroundAndSignalHist(var):

    FullData   = formatForTagging([var], signalPath=fileNamesCSV["pipi"], backgroundPath=fileNamesCSV["kpisb"], evenSplit=True) 
    signal     = FullData[FullData["isSignal"] == 1.0]
    background = FullData[FullData["isSignal"] == 0.0]

    plt.figure(figsize=(10, 8))
    plt.hist(             [FullData[var],       signal[var],       background[var]], 
               weights  = [FullData["weights"], signal["weights"], background["weights"]],
               label    = ["Sum",               "Signal",          "Background"],
               color    = [colors["black"],     colors["red"],     colors["blue"]],
               bins     = 100,
               histtype = "step")
    
    plt.title(var)
    plt.legend()
    plt.savefig(f"{imagePath}SeperationHists/{var}_Seperation.png", dpi=227)
    plt.close()


def createAllBackgroundAndSignalHists():
    allVariables = sharedVariables + uniqueVariables["any"]
    for var in allVariables:
        createBackgroundAndSignalHist(var)


if __name__ == "__main__":
    createAllBackgroundAndSignalHists()
    



