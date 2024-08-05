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

#DONE
def loadModels(name = "Bestpipi"):
    RF = joblib.load(f"savedModels/{name}/RF{name}.joblib")
    AD = joblib.load(f"savedModels/{name}/AD{name}.joblib")
    GB = joblib.load(f"savedModels/{name}/GB{name}.joblib")
    NN = joblib.load(f"savedModels/{name}/NN{name}.joblib")
    return (RF,AD,GB,NN)

#DONE
def plotROCcurve(models, inputData = pd.DataFrame(), cut = None):

    plt.plot([0,1],[0,1], c="black", linestyle=":")
    plt.grid(linestyle="--",alpha=0.3)
    plt.title("ROC curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postive Rate")

    for model in models:
        (TPR, FPR) = model.getROCCurveValues(inputData)
        plt.plot(TPR, FPR, label=f"{model.abbreviation}")
    
    plt.legend()
    plt.show()

#DONE
def plotSeperatedPredictions(model, inputData = pd.DataFrame()):

    (SignalData, BackgroundData) = model.getSeperatedPredictions(inputData)

    plt.title(f"{model.name} Model Prediction Distribution")
    plt.hist( [SignalData[f"{model.name} Prediction"], BackgroundData[f"{model.name} Prediction"]],
    weights = [SignalData["weights"],    BackgroundData["weights"]   ],
    label   = ["Signal",                 "Background"],
    **histStyle)

    plt.legend()
    plt.show()

#DONE
def plotNewSignal(model, variable, inputData = pd.DataFrame()):

    plt.title(f"{model.name} Model Prediction Distribution")
    if inputData.empty: 
        inputData = model.X_test
        inputData["weights"] = model.W_test
    
    FilteredData = inputData[model.predict(inputData)>0.5]

    plt.hist( [inputData[variable],  FilteredData[variable]] ,
    weights = [inputData["weights"], FilteredData["weights"]],
    label   = ["Oringinal",            "Background Removed"],
    bins    = 50, 
    alpha   = 0.8, 
    histtype = "step")

    plt.legend()
    plt.show()

#DONE
def plotFigureOfMerit(model, inputData = pd.DataFrame()):
    FoM = model.getFigureOfMerit(inputData)
    fig, ax1 = plt.subplots()
    ax1.set_title(f"{model.name}")
    ax1.plot(np.linspace(0,1,1001), FoM, c = colors["red"], label="Figure of Merit")
    ax1.set_xlabel("Cut")
    ax1.set_ylabel("Figure of Merit", color=colors["red"])
    ax1.set_xlim(0,1)
    ax1.set_ylim(0)

    bestCut = model.getBestCut(inputData)
    print("Best Cut: ", bestCut)
    plt.axvline(x = bestCut, c = colors["black"], linestyle=":", label=f"Best Cut: {bestCut}", alpha=0.2)
    plt.show()

#DONE
def plotEfficiencies(model, inputData = pd.DataFrame()):
    (sigEff, bacEff) = model.getEfficiencies(inputData)
    bestCut = model.getBestCut(inputData)
    print("Best Cut: ", bestCut)

    plt.plot(np.linspace(0,1,1001),sigEff, label="Signal", alpha = 0.2)
    plt.plot(np.linspace(0,1,1001),bacEff, label="Background", alpha = 0.2)
    plt.plot(np.linspace(0,1,1001),np.array(sigEff) - np.array(bacEff), label="Diff")
    plt.axvline(x = bestCut, c = colors["black"], linestyle=":", label=f"Best Cut", alpha=0.2)
    plt.legend()
    plt.show()

#DONE
def plotVariableDistribution(variable, model, inputData = pd.DataFrame(), cut = 0.5):

    if inputData.empty:
        inputData = model.X_test
        inputData["isSignal"] = model.Y_test
        inputData["weights"]  = model.W_test

    X_Data = inputData[model.inputVariables]
    inputData["Prediction"] = roundUp(model.predict(X_Data), cut)

    SignalData     = inputData[inputData["Prediction"] == 1]
    BackgroundData = inputData[inputData["Prediction"] == 0]

    plt.title(f"{variable} Prediction Distribution")
    plt.hist( [SignalData[variable], BackgroundData[variable],  inputData[variable]], 
    weights = [SignalData["weights"],BackgroundData["weights"], inputData["weights"]],
    label   = ["Signal","Background","Total"], 
    bins=50, 
    histtype="step")
    plt.legend()
    plt.show()

#DONE
def plotCorelationMatrix():

    plt.title("Correlation Matrix")
    data   = pd.read_csv("data/Raw/EvenTestData.csv", index_col=0)
    X_data = data[fullVariables["any"] + ["isSignal"]]

    print(X_data.corr())
    dataplot = sb.heatmap(X_data.corr())

    plt.show()

#DONE
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
    plotVariableDistribution("nTracks",GB)

