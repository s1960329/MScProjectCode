import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import seaborn as sb

from misc import *

from scriptsPython.Classifiers import ForestClassifiers, NeuralNetworkClassifier

histStyle         = {"bins"     : 50, 
                     "alpha"    : 0.8, 
                     "histtype" : "step"}


def createProbabilityDistribution(model, data):
    SignalData     = data[data["isSignal"] == 1.0]
    BackgroundData = data[data["isSignal"] == 0.0]

    X_SignalData     = SignalData[model.inputVariables]
    X_BackgroundData = BackgroundData[model.inputVariables]

    SignalData["Prediction"]     = model.predict(X_SignalData)
    BackgroundData["Prediction"] = model.predict(X_BackgroundData)
    
    plt.title("Model Prediction Distribution")
    plt.hist( [SignalData["Prediction"], BackgroundData["Prediction"]],
    weights = [SignalData["weights"],    BackgroundData["weights"]   ],
    label   = ["Signal",                 "Background"],
    **histStyle)
    plt.legend()

    plt.show()



def getFOMandEfflist(model, data, cuts):
    sigEff = []
    l      = []

    data["Prediction"] = model.predict(data)

    SignalData     = data[data["isSignal"] == 1.0]
    BackgroundData = data[data["isSignal"] == 0.0]
    
    for cut in cuts:
        eBDT  = len(SignalData[ (SignalData["Prediction"] > cut) ]) / len(SignalData)
        Bcomb = len(data[ (data["Prediction"] > cut) ])
        l.append(len(SignalData)*eBDT / np.sqrt(len(SignalData)*eBDT + Bcomb))
        sigEff.append(eBDT)

    return (sigEff, l)



def createFigureOfMerit(model, data):

    cuts = np.linspace(0,1,1000)
    (sigEff, l) = getFOMandEfflist(model, data, cuts)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_title(f"{model.name}")
    ax1.plot(cuts, l,      c = colors["red"],  label="Figure of Merit")
    ax2.plot(cuts, sigEff, c = colors["blue"], label="Signal Efficiency")

    ax1.set_xlabel("Cut")
    ax1.set_ylabel("Figure of Merit"  , color=colors["red"])
    ax2.set_ylabel("Signal Efficiency", color=colors["blue"])

    ax1.set_xlim(0,1)
    ax1.set_ylim(0)
    ax2.set_ylim(0)

    plt.show()



def variableDistribution(variable, model, data):
    X_Data = data[model.inputVariables]
    data["Prediction"] = roundUp(model.predict(X_Data))

    SignalData     = data[data["Prediction"] == 1]
    BackgroundData = data[data["Prediction"] == 0]

    plt.title(f"{variable} Prediction Distribution")
    plt.hist( [SignalData[variable], BackgroundData[variable] ],
    weights = [SignalData["weights"],BackgroundData["weights"]],
    label   = ["Signal",            "Background"],
    **histStyle)
    plt.legend()

    plt.show()


def createCorelationMatrix():

    plt.title("Correlation Matrix")
    data   = pd.read_csv("data/Raw/EvenTestData.csv", index_col=0)
    X_data = data[fullVariables["any"] + ["isSignal"]]

    print(X_data.corr())
    dataplot = sb.heatmap(X_data.corr())

    plt.show()

def CreateCorelationPlots():
    
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

    model  = joblib.load("savedModels/EvenFinalModel/GBEvenFinalModel.joblib")
    data   = pd.read_csv("data/Raw/EvenTestData.csv", index_col=0)

    # print(model.predict(X_test))
    # createCorelationMatrix()
    # variableDistribution("nTracks", model, data)
    # CreateCorelationPlots()
    # createFigureOfMerit(model, data)
    createProbabilityDistribution(model, data)