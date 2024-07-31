import uproot
import joblib

import matplotlib.pyplot       as plt
import numpy                   as np
import pandas                  as pd

from   sklearn.metrics         import roc_curve, roc_auc_score
from   hep_ml                  import reweight
from   sklearn.model_selection import train_test_split
from   tensorflow.keras.models import load_model  # type: ignore
from tensorflow import keras


dataPath          = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/data"
imagePath         = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs"
modelPath         = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/savedModels"


sharedVariables   = ["nTracks", "B_P", "gamma_PT", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV"]
VariableUnits     = ["",        "(MeV/c)","(MeV/c)",   "",                "",      "",             "",                            "(mm)",         "(MeV/c)",       "(μm)", "(MeV/c)","(μm)","(MeV/c)","(μm)"]


uniqueVariables   = {"kpi"      : ["Kst_892_0_PT",        "Kst_892_0_IP_OWNPV",        "Kplus_PT",       "Kplus_IP_OWNPV" ],
                     "pipi"     : ["rho_770_0_PT",        "rho_770_0_IP_OWNPV",        "piplus_PT",      "piplus_IP_OWNPV"],
                     "any"      : ["daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]}


fullVariables     = {"kpi"      : sharedVariables + uniqueVariables["kpi"],
                     "pipi"     : sharedVariables + uniqueVariables["pipi"],
                     "any"      : sharedVariables + uniqueVariables["any"]}


UnitDict = dict(zip(fullVariables["any"], VariableUnits))


fileNamesRoot     = {"kpi"      : f"{dataPath}/dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "kpisb"    : f"{dataPath}/dataROOT/kpiG_sideband_2018.root",
                     "pipi"     : f"{dataPath}/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "sm"       : f"{dataPath}/dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root",
                     "f1"       : f"{dataPath}/dataROOT/BDT_2018_fold1.root",
                     "f2"       : f"{dataPath}/dataROOT/BDT_2018_fold2.root"}


fileNamesCSV      = {"kpi"      : f"{dataPath}/dataCSV/kpi_montecarlo_reweighted.csv",
                     "pipi"     : f"{dataPath}/dataCSV/pipi_montecarlo_reweighted.csv",
                     "kpisb"    : f"{dataPath}/dataCSV/kpi_sideband.csv",
                     "sm"       : f"{dataPath}/dataCSV/kpi_sample.csv",
                     "f1"       : f"{dataPath}/dataCSV/kpi_BDT_2018_fold1.csv",
                     "f2"       : f"{dataPath}/dataCSV/kpi_BDT_2018_fold2.csv",
                     "f1v"      : f"{dataPath}/dataCSV/kpi_BDT_2018_fold1_test.csv",
                     "f1t"      : f"{dataPath}/dataCSV/kpi_BDT_2018_fold1_train.csv",
                     "f2v"      : f"{dataPath}/dataCSV/kpi_BDT_2018_fold2_test.csv",
                     "f2t"      : f"{dataPath}/dataCSV/kpi_BDT_2018_fold2_train.csv"}


histStyle         = {"bins"     : 50, 
                     "density"  : True,
                     "alpha"    : 0.8, 
                     "histtype" : "step"}


colors            = {"red"      : "#d80645", 
                     "blue"     : "#006cd4",  
                     "green"    : "#2bad6a",
                     "yellow"   : "#ffd080",
                     "purple"   : "#c3608d",
                     "orange"   : "#ff9d0a",
                     "black"    : "#000000"}


#Reads a root file and converts it's data to a pandas array
def readRootFile(rootFilePath, variables):
    with uproot.open(rootFilePath) as TChain: # type: ignore
        TTree = TChain["DecayTree"]

    rootFile = TTree.arrays(variables, library="pd", cut = "(abs(B_M01-895.55)<100)", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
    rootFile = rootFile.reset_index(drop=True)
    return rootFile


#Converts the contents of the root file to a csv file
def rootToCSV(decayMode="sm", variables=fullVariables["kpi"] + ["NB0_Kpigamma_sw"]):
    
    LoadedData = readRootFile(fileNamesRoot[decayMode], variables)
    LoadedData.to_csv(fileNamesCSV[decayMode])
    print(f"Data saved to csv file {fileNamesCSV[decayMode]}")


def createAllReweightedHistograms():
    allVariables = sharedVariables + uniqueVariables["any"]
    for var in allVariables:
        createReweightedHistogram(var)


#Creates a reweighted histogram with a correponding plot
def createReweightedHistogram(variable = "nTracks", decayMode = "pipi"):
    
    canvas, (disti,ratio) = plt.subplots(2,1, gridspec_kw={"height_ratios" : [2,1] },figsize=(8, 7))
    canvas.suptitle(f"{variable} Distribution {UnitDict[variable]}")

    MonteCarloData = pd.read_csv(fileNamesCSV[decayMode], index_col=0)
    SampleData     = pd.read_csv(fileNamesCSV["sm"],   index_col=0)

    kpiDict  = dict(zip(uniqueVariables["kpi" ],uniqueVariables["any"]))
    pipiDict = dict(zip(uniqueVariables["pipi"],uniqueVariables["any"]))

    SampleData     = SampleData.rename(    columns = kpiDict )
    MonteCarloData = MonteCarloData.rename(columns = pipiDict)

    minX = 0    #max(min(SampleData[variable]), min(MonteCarloData[variable]))
    maxX = 10   #min(max(SampleData[variable]), max(MonteCarloData[variable]))
    print(variable, minX, maxX)

    histY, histX, _  = disti.hist([SampleData[variable],         MonteCarloData[variable],  MonteCarloData[variable]], 
                       weights =  [SampleData["NB0_Kpigamma_sw"],MonteCarloData["weights"], np.ones(len(MonteCarloData))],
                       label   =  ["Sample",                     "Weighted Monte Carlo",    "Monte Carlo"],
                       range   =  [minX, maxX],
                       color   =  [colors["red"],                colors["green"],           colors["blue"]],
                       **histStyle)

    histBars = dict( zip(["sm","wmc", "mc"],histY) )

    ratioDataMC  = ((histBars["sm"] - histBars["mc" ])**2) / histBars["sm"]
    ratioDataMCW = ((histBars["sm"] - histBars["wmc"])**2) / histBars["sm"]
    ratioDataMC[    (histBars["sm"] < 0) | (histBars["mc" ] < 0)] = np.nan
    ratioDataMCW[   (histBars["sm"] < 0) | (histBars["wmc"] < 0)] = np.nan
    

    ratio.plot(histX[:-1], ratioDataMC,  c=colors["blue"],  alpha=0.8, linewidth=0.7)
    ratio.plot(histX[:-1], ratioDataMCW, c=colors["green"], alpha=0.8, linewidth=0.7)
    ratio.set_title(f"Chi Squared Difference from Sample, Total Change: {float("%.3g" % sum(np.nan_to_num(ratioDataMCW - ratioDataMC)))}")
    ratio.set_xlim(min(histX), max(histX))
    ratio.grid(axis="both", linestyle="dashed", alpha=0.3)

    disti.set_xlim(min(histX), max(histX))
    disti.xaxis.set_tick_params(which = "both",labelbottom=False, bottom=False)
    disti.grid(axis="both", linestyle="dashed", alpha=0.3)
    disti.legend()

    plt.savefig(f"{imagePath}/Reweighted/{variable}_{decayMode}_ReweightedDistribution.png", dpi=227)
    plt.close()


#Imports signal and background data and shuffles them into a single dataframe
def loadShuffledData(variables, signalPath, backgroundPath, evenSplit):

    signalDict     = dict(zip(uniqueVariables["pipi"],uniqueVariables["any"]))
    backgroundDict = dict(zip(uniqueVariables["kpi" ],uniqueVariables["any"]))

    signal         = pd.read_csv(signalPath, index_col=0)
    signal         = signal.rename(columns = signalDict)

    background     = pd.read_csv(backgroundPath, index_col=0)
    background     = background.rename(columns = backgroundDict)

    if evenSplit:
        minSize    = min(len(signal),len(background))
        signal     = signal.sample(n=minSize)
        background = background.sample(n=minSize)

    signal["isSignal"]     = np.array([1.0]* len(signal))
    background["isSignal"] = np.array([0.0]* len(background))  
    background["weights" ] = np.ones(len(background))  

    variables = variables + ["isSignal","weights"]
    FullData  = pd.concat([signal[variables], background[variables]])
    FullData  = FullData.sample(frac=1)
    FullData  = FullData.reset_index(drop=True)

    return (FullData,signal,background)


#Smoothly splits the shuffed data to 
def SplitDataForTraining(variables, signalPath = fileNamesCSV["pipi"], backgroundPath=fileNamesCSV["kpisb"], evenSplit=True):
    
    (FullData, signal, background) = loadShuffledData(variables, signalPath, backgroundPath, evenSplit)
    
    train, test = train_test_split(FullData)

    X_train = train[variables]
    Y_train = train["isSignal"]
    W_train = train["weights"]

    X_test  = test[variables]
    Y_test  = test["isSignal"]
    W_test  = test["weights"]

    return ((X_train,Y_train,W_train),(X_test,Y_test,W_test))


#Normalises the data between zero and one for NN training
def normaliseData(data, varToNorm):
    for var in varToNorm:
        data[var] = (data[var] - min(data[var])) / (max(data[var])  -  min(data[var]) )

    return data


def createTreeROCcurve(classifier, X_test, Y_test):
    Y_pred = classifier.predict_proba(X_test)[:, 1]
    falsePositiveRate, truePositiveRate, threshold = roc_curve(Y_test, Y_pred)
    aucScore = roc_auc_score(Y_test, Y_pred)
    return falsePositiveRate, truePositiveRate, aucScore

#Takes the NN summary and converts it to a string for exporting
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
    plt.savefig(imagePath+imageName)
    plt.close()







def loadModels(name = "testModel"):
    AD = joblib.load(f"{modelPath}{name}/AD{name}.joblib")
    NN = joblib.load(f"{modelPath}{name}/GB{name}.joblib")
    RF = joblib.load(f"{modelPath}{name}/RF{name}.joblib")
    GB = joblib.load(f"{modelPath}{name}/NN{name}.keras")
    return (AD,GB,NN,RF)





if __name__ == "__main__":

    # "piminus_IP_OWNPV"
    # "daughterplus_IP_OWNPV"
    # "B_SmallestDeltaChi2OneTrack"
    createReweightedHistogram(variable = "piminus_IP_OWNPV", decayMode = "pipi")
    createReweightedHistogram(variable = "daughterplus_IP_OWNPV", decayMode = "pipi")

    # FullDataNorm = pd.read_csv(f"{dataPath}/dataLearn/FullData.csv", index_col=0)

    # FullDataNorm   = normaliseData(FullDataNorm, varToNorm = fullVariables["any"])
    # backgroundNorm = FullDataNorm[FullDataNorm["isSignal"] == 0.0]
    # signalNorm     = FullDataNorm[FullDataNorm["isSignal"] == 1.0]
    # trainDataNorm, testDataNorm    = train_test_split(FullDataNorm)

    # FullDataNorm.to_csv(f"{dataPath}/dataLearn/FullDataNorm.csv")
    # trainDataNorm.to_csv(f"{dataPath}/dataLearn/TestNorm.csv")
    # testDataNorm.to_csv(f"{dataPath}/dataLearn/TrainNorm.csv")
    # backgroundNorm.to_csv(f"{dataPath}/dataLearn/BackgroundNorm.csv")
    # signalNorm.to_csv(f"{dataPath}/dataLearn/SignalNorm.csv")



