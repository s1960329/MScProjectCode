from sklearn.metrics         import roc_curve, roc_auc_score
from hep_ml                  import reweight
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import uproot

histStyle         = {"bins" : 100, "density" : True, "alpha" : 1, "histtype" : "step"}

colors            = {"red"    : "#d80645", 
                     "blue"   : "#006cd4",  
                     "green"  : "#2bad6a",
                     "yellow" : "#ffd080",
                     "purple" : "#c3608d",
                     "orange" : "#ff9d0a",
                     "black"  : "#000000"}

dataPath          = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/data/"
imagePath         = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/"

sharedVariables   = ["nTracks", "B_P", "gamma_PT", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV"]


uniqueVariables   = {"kpi" : ["Kst_892_0_PT",        "Kst_892_0_IP_OWNPV",        "Kplus_PT",       "Kplus_IP_OWNPV" ],
                     "pipi": ["rho_770_0_PT",        "rho_770_0_IP_OWNPV",        "piplus_PT",      "piplus_IP_OWNPV"],
                     "any" : ["daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]}

fullVariables     = {"kpi"  : sharedVariables + uniqueVariables["kpi"],
                     "pipi" : sharedVariables + uniqueVariables["pipi"],
                     "any"  : sharedVariables + uniqueVariables["any"]}

fileNamesRoot     = {"kpi"  : f"{dataPath}dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "kpisb": f"{dataPath}dataROOT/kpiG_sideband_2018.root",
                     "pipi" : f"{dataPath}dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "sm"   : f"{dataPath}dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root"}

fileNamesCSV      = {"kpi"  : f"{dataPath}dataCSV/kpi_montecarlo_reweighted.csv",
                     "pipi" : f"{dataPath}dataCSV/pipi_montecarlo_reweighted.csv",
                     "kpisb": f"{dataPath}dataCSV/kpi_sideband.csv",
                     "sm"   : f"{dataPath}dataCSV/kpi_sample.csv"}


def createHistogram(variable = "nTracks", decayMode = "pipi"):
    
    canvas, (disti,ratio) = plt.subplots(2,1, gridspec_kw={"height_ratios" : [2,1] },figsize=(8, 7))
    canvas.suptitle(f"{variable} Distribution")
    canvas.tight_layout()

    MonteCarloData = pd.read_csv(fileNamesCSV[decayMode], index_col=0)
    SampleData     = pd.read_csv(fileNamesCSV["sm"],   index_col=0)


    histY, histX, _  = disti.hist([SampleData[variable],         MonteCarloData[variable],  MonteCarloData[variable]], 
                       weights =  [SampleData["NB0_Kpigamma_sw"],MonteCarloData["weights"], np.ones(len(MonteCarloData))],
                       label   =  ["Sample",                     "Weighted Monte Carlo",    "Monte Carlo"],
                       color   =  [colors["red"],                colors["green"],           colors["blue"]],
                       **histStyle)
    
    histBars = dict( zip(["sm","wmc", "mc"],histY) )


    ratioDataMC  = ((histBars["sm"] - histBars["mc" ])**2) / histBars["sm"]
    ratioDataMCW = ((histBars["sm"] - histBars["wmc"])**2) / histBars["sm"]

    ratio.scatter(histX[:-1],ratioDataMC,  c=colors["blue"], s=5)
    ratio.scatter(histX[:-1],ratioDataMCW, c=colors["green"], s=5)
    ratio.set_title(f"Chi Squared Difference: {sum(np.nan_to_num(ratioDataMCW - ratioDataMC))}")
    ratio.set_xlim(min(histX), max(histX))
    ratio.grid(axis="both", linestyle="dashed", alpha=0.3)

    disti.set_xlim(min(histX), max(histX))
    disti.xaxis.set_tick_params(which = "both",labelbottom=False, bottom=False)
    disti.grid(axis="both", linestyle="dashed", alpha=0.3)
    disti.legend()

    plt.savefig(f"{imagePath}{variable}_{decayMode}_ReweightedDistribution.png", dpi=227)
    plt.close()


def formatForClassification(trainingVariables, signalPath=fileNamesCSV["pipi"], backgroundPath=fileNamesCSV["kpisb"], evenSplit=False):
    signalDict = dict(zip(uniqueVariables["pipi"],uniqueVariables["any"]))
    signalData = pd.read_csv(signalPath, index_col=0)
    signalData = signalData.rename(columns = signalDict)
    signalData["isSignal"] = np.ones(len(signalData))

    backgroundDict = dict(zip(uniqueVariables["kpi"],uniqueVariables["any"]))
    backgroundData = pd.read_csv(backgroundPath, index_col=0)
    backgroundData = backgroundData.rename(columns = backgroundDict)
    backgroundData["weights"]  = np.ones(len(backgroundData))
    backgroundData["isSignal"] = np.zeros(len(backgroundData))


    if evenSplit:
        minSize        = min(len(signalData),len(backgroundData))
        backgroundData = backgroundData.sample(n=minSize)
        signalData     = signalData.sample(n=minSize)


    FullData = pd.concat([backgroundData, signalData])

    train, test = train_test_split(FullData)

    X_train = train[trainingVariables]
    Y_train = train["isSignal"]
    W_train = train["weights"]

    X_test  = test[trainingVariables]
    Y_test  = test["isSignal"]
    W_test  = test["weights"]

    return ((X_train,Y_train,W_train),(X_test,Y_test,W_test))


def normaliseData(data):
    for var in data.columns:
        data[var] = (data[var] - min(data[var])) / (max(data[var])  -  min(data[var]) )

    return data

def readRootFile(rootFilePath, variables):
    with uproot.open(rootFilePath) as TChain: # type: ignore
        TTree = TChain["DecayTree"]

    rootFile = TTree.arrays(variables, library="pd", cut = "(abs(B_M01-895.55)<100)", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
    rootFile = rootFile.reset_index(drop=True)
    return rootFile


def rootToCSV(decayMode = "sm", variables = fullVariables["kpi"] + ["NB0_Kpigamma_sw"]):
    
    LoadedData = readRootFile(fileNamesRoot[decayMode], variables)
    LoadedData.to_csv(fileNamesCSV[decayMode])
    print(f"Data saved to csv file {fileNamesCSV[decayMode]}")


def createTreeROCcurve(classifier, X_test, Y_test):
    Y_pred = classifier.predict_proba(X_test)[:, 1]
    falsePositiveRate, truePositiveRate, threshold = roc_curve(Y_test, Y_pred)
    aucScore = roc_auc_score(Y_test, Y_pred)
    return falsePositiveRate, truePositiveRate, aucScore


def NNSummaryToSring(NNmodel):
    stringlist = []
    NNmodel.summary(print_fn=lambda x: stringlist.append(x))
    modelSummary = "\n".join(stringlist)
    
    return modelSummary


def roundUp(data, cutOff=0.5):

    predictions = []

    for prob in data:
        if prob > cutOff: prediction = 1
        else: prediction = 0
        predictions.append(float(prediction))

    return np.array(predictions)

from tensorflow import keras

def FindBestCutOff(X_testNormalised, Y_test, imageName="CutOffplot.png"):

    
    scores  = []
    cutOffs = []
    NNmodel = keras.models.load_model("savedModels/OverfittingTest_5_BestNN/NNmodel_OverfittingTest_5.keras")

    predictions = NNmodel.predict(X_testNormalised).flatten()
    for cutOff in np.linspace(0, 1, 1001):
        predictionsWithCutOff = roundUp(predictions, cutOff)
        score = roc_auc_score(Y_test,  predictionsWithCutOff)
        scores.append(score)
        cutOffs.append(cutOff)

    bestCutOff = cutOffs[scores.index(max(scores))]

    plt.title(f"Optimal Cut Off: {np.round(bestCutOff,3)}")
    plt.plot(cutOffs, scores)
    plt.savefig(imagePath+imageName, dpi=227)
    plt.close()

if __name__ == "__main__":
    createHistogram()


