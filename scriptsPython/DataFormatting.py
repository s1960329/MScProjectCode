
import uproot
import pandas as pd
import ROOT
import numpy  as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from HistogramReweighter import GBReweight


dataPath          = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/data"

fileNamesRoot     = {"kpi"      : f"{dataPath}/dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "kpisb"    : f"{dataPath}/dataROOT/kpiG_sideband_2018.root",
                     "pipisb"   : f"{dataPath}/dataROOT/pipiG_sideband_2018_rhoMass.root",
                     "pipi"     : f"{dataPath}/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "sm"       : f"{dataPath}/dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root",
                     "f1"       : f"{dataPath}/dataROOT/BDT_2018_fold1.root",
                     "f2"       : f"{dataPath}/dataROOT/BDT_2018_fold2.root"}

sharedVariables   = ["nTracks", "B_P", "gamma_PT", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV"]

uniqueVariables   = {"kpi"      : ["Kst_892_0_PT",        "Kst_892_0_IP_OWNPV",        "Kplus_PT",       "Kplus_IP_OWNPV" ],
                     "pipi"     : ["rho_770_0_PT",        "rho_770_0_IP_OWNPV",        "piplus_PT",      "piplus_IP_OWNPV"],
                     "any"      : ["daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]}

fullVariables     = {"kpi"      : sharedVariables + uniqueVariables["kpi"],
                     "pipi"     : sharedVariables + uniqueVariables["pipi"],
                     "any"      : sharedVariables + uniqueVariables["any"]}


def readRootFile(rootFilePath, variables):
    with uproot.open(rootFilePath) as TChain: # type: ignore
        TTree = TChain["DecayTree"]

    rootFile = TTree.arrays(variables, library="pd", cut = "(abs(B_M01-895.55)<100)", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
    rootFile = rootFile.reset_index(drop=True)
    return rootFile


def rootToCSV(rootFilepath, csvFilePath, variables=fullVariables["kpi"] + ["NB0_Kpigamma_sw"]):
    
    LoadedData = readRootFile(rootFilepath, variables)
    LoadedData.to_csv(csvFilePath)
    print(f"Data saved to csv file {csvFilePath}")


def createSignalAndBackground():

    rootToCSV(rootFilepath = f"{dataPath}/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root", 
              csvFilePath  = f"data/Raw/SignalData.csv", 
              variables    = fullVariables["pipi"])
    
    pipi = GBReweight(decayMode="pipi")
    pipi.executeAll()


def combineSignalAndBackground(mode = "pipi"):

    BackgrData = pd.read_csv(f"data/{mode}/BackgroundData.csv", index_col = 0)
    SignalData = pd.read_csv(f"data/{mode}/SignalData.csv",     index_col = 0)

    VarDict    = dict(zip(uniqueVariables[mode],uniqueVariables["any"]))

    BackgrData = BackgrData.rename(columns = VarDict)
    SignalData = SignalData.rename(columns = VarDict)

    BackgrData["weights"]  = np.ones(len(BackgrData))

    BackgrData["isSignal"] = np.zeros(len(BackgrData))
    SignalData["isSignal"] = np.ones( len(SignalData))

    FullData = pd.concat([BackgrData, SignalData])
    FullData = FullData.sample(frac=1)
    FullData = FullData.reset_index(drop=True)
    
    FullData.to_csv(f"data/{mode}/FullData.csv")


def createTestAndTrainData(mode = "pipi"):

    FullData = pd.read_csv(f"data/{mode}/FullData.csv", index_col = 0)
    train, test = train_test_split(FullData)

    test. to_csv(f"data/{mode}/TestData.csv")
    train.to_csv(f"data/{mode}/TrainData.csv")
   

def splitSignalEvenly(filePath):
    Data = pd.read_csv(filePath, index_col=0)

    SampleSize = min( len(Data[Data["isSignal"] == 0]), len(Data[Data["isSignal"] == 1]) )
    EvenData = Data.groupby("isSignal", as_index=False, group_keys=False).apply(lambda x: x.sample(SampleSize))
    EvenData = EvenData.sample(frac=1)
    
    return EvenData

def createEvenTestAndTrainData(mode = "pipi"):
    
    testEven  = splitSignalEvenly(f"data/{mode}/TestData.csv")
    testEven. to_csv(f"data/{mode}/EvenTestData.csv")
    
    trainEven = splitSignalEvenly(f"data/{mode}/TrainData.csv")
    trainEven.to_csv(f"data/{mode}/EvenTrainData.csv")
   

def createLaisData():
    fold = "fold2"
    BDT2018name = "../data/dataROOT/BDT_2018_fold1.root"
    file = uproot.open(BDT2018name)

    print(file.keys())
    # rootFile = TTree.arrays(variables, library="pd", cut = "(abs(B_M01-895.55)<100)", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
    # rootFile = rootFile.reset_index(drop=True)
    # return rootFile

    # ROClais = file["dataset/TestTree"].values()
    # ROClaisEdges = file["dataset/Method_BDT/BDT_2018_fold2/MVA_BDT_2018_fold2_rejBvsS"].axis().edges()



    

if __name__ == "__main__":
    var      = "gamma_PT"
    sideband = readRootFile(rootFilePath=fileNamesRoot["pipisb"], variables=fullVariables["pipi"])
    signal   = readRootFile(rootFilePath=fileNamesRoot["pipi"  ], variables=fullVariables["pipi"])
    full     = pd.concat((signal,sideband))

    plt.hist([signal[var],sideband[var], full[var]], label=["Signal","Background","Total"], bins=50, histtype="step")
    plt.title(f"{var} Distribution")
    plt.legend()
    plt.show()
