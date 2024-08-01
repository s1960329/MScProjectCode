from xml.parsers.expat import model
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import seaborn as sb

from Classifiers import *


histStyle         = {"bins"     : 50, 
                     "alpha"    : 0.8,
                     "density"  : True, 
                     "histtype" : "step"}

def loadModels(name = "Bestpipi"):
    RF = joblib.load(f"savedModels/{name}/RF{name}.joblib")
    AD = joblib.load(f"savedModels/{name}/AD{name}.joblib")
    GB = joblib.load(f"savedModels/{name}/GB{name}.joblib")
    NN = joblib.load(f"savedModels/{name}/NN{name}.joblib")
    return (RF,AD,GB,NN)

#TODO plot the ROC curve that you get from the model
def plotROCcurve(model, inputData = pd.DataFrame(), cut = None):
    pass

#TODO test seperated predictions
def plotSeperatedPredictions(model, inputData = pd.DataFrame()):

    (SignalData,BackgroundData) = model.getSeperatedPredictions(inputData)

    plt.title(f"{model.name} Model Prediction Distribution")
    plt.hist( [SignalData["Prediction"], BackgroundData["Prediction"]],
    weights = [SignalData["weights"],    BackgroundData["weights"]   ],
    label   = ["Signal",                 "Background"],
    **histStyle)
    plt.legend()
    plt.show()

#TODO adapt for new code
def plotFigureOfMerit(model, inputData = pd.DataFrame()):

    (sigEff, bacEff, FoM) = model(model, data)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_title(f"{model.name}")
    ax1.plot(np.linspace(0,1,1001), FoM,      c = colors["red"],   label="Figure of Merit")
    ax2.plot(np.linspace(0,1,1001), sigEff,   c = colors["blue"],  label="Signal Efficiency")

    ax1.set_xlabel("Cut")
    ax1.set_ylabel("Figure of Merit"  , color=colors["red"])
    ax2.set_ylabel("Signal Efficiency", color=colors["blue"])

    ax1.set_xlim(0,1)
    ax1.set_ylim(0)
    ax2.set_ylim(0)

    plt.show()

#TODO adapt for new code
def plotEfficiencies(model, inputData = pd.DataFrame()):
    (sigEff, bacEff, FoM) = getFOMandEfflist(model, data)

    bestCut = FindBestCut(model, data)
    print("Best Cut: ", bestCut)

    plt.plot(np.linspace(0,1,1001),sigEff, label="Signal Efficiency", alpha = 0.2)
    plt.plot(np.linspace(0,1,1001),bacEff, label="Background Efficiency", alpha = 0.2)
    plt.plot(np.linspace(0,1,1001),np.array(sigEff) - np.array(bacEff), label="Difference")
    plt.axvline(x = bestCut, c = colors["black"], label=f"Best Cut: {bestCut}", alpha=0.2)
    plt.legend()
    plt.show()

#TODO add the true distributions for comparision
def plotVariableDistribution(variable, model, data = pd.DataFrame.empty, cut = 0.2):

    if data == pd.DataFrame.empty:
        data = model.X_test
        data["isSignal"] = model.Y_test
        data["weights"]  = model.W_test

    X_Data = data[model.inputVariables]
    data["Prediction"] = roundUp(model.predict(X_Data), cut)

    SignalData     = data[data["Prediction"] == 1]
    BackgroundData = data[data["Prediction"] == 0]

    plt.title(f"{variable} Prediction Distribution")
    plt.hist( [SignalData[variable], BackgroundData[variable] ],
    weights = [SignalData["weights"],BackgroundData["weights"]],
    label   = ["Signal",            "Background"],
    **histStyle)
    plt.legend()
    plt.show()

#TODO?
def plotCorelationMatrix():

    plt.title("Correlation Matrix")
    data   = pd.read_csv("data/Raw/EvenTestData.csv", index_col=0)
    X_data = data[fullVariables["any"] + ["isSignal"]]

    print(X_data.corr())
    dataplot = sb.heatmap(X_data.corr())

    plt.show()

#TODO?
def createCorelationPlots():
    
    plt.title("Correlation Matrix")
    data   = pd.read_csv("data/Raw/EvenTestData.csv", index_col=0)
    data = data[fullVariables["any"]+["isSignal"]]

    pairplot = sb.pairplot(data, hue="isSignal")
    pairplot.map_upper(sb.histplot)
    pairplot.map_lower(sb.kdeplot, fill=True)
    pairplot.map_diag(sb.histplot, kde=True)

    plt.show()
    pass


if __name__ == "__main__":

    (RF,AD,GB,NN) = loadModels()
    (TPR, FPR) = NN.createROCcurve()

    plt.plot(TPR, FPR)
    plt.show()