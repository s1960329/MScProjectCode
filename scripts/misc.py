
import uproot
import pandas as pd
import numpy  as np


dataPath         = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/data"
imagePath        = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs"
modelPath        = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/savedModels"

sharedFeatures   = ["nTracks", "B_P", "gamma_PT", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV"]

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

fileNamesRoot     = {"kpi"      : f"{dataPath}/dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "kpisb"    : f"{dataPath}/dataROOT/kpiG_sideband_2018.root",
                     "pipi"     : f"{dataPath}/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root",
                     "pipisb"   : f"{dataPath}/dataROOT/pipiG_sideband_2018_rhoMass.root",
                     "sm"       : f"{dataPath}/dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root"}

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

uniqueFeatures   = {"kpi"       : ["Kst_892_0_PT",        "Kst_892_0_IP_OWNPV",        "Kplus_PT",       "Kplus_IP_OWNPV" ],
                     "pipi"     : ["rho_770_0_PT",        "rho_770_0_IP_OWNPV",        "piplus_PT",      "piplus_IP_OWNPV"],
                     "any"      : ["daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]}

allFeatures     = {"kpi"        : sharedFeatures + uniqueFeatures["kpi"],
                     "pipi"     : sharedFeatures + uniqueFeatures["pipi"],
                     "any"      : sharedFeatures + uniqueFeatures["any"]}

featureUnits     = ["", "MeV/c","MeV/c", "", "", "", "", "mm", "MeV/c", "μm", "MeV/c", "μm", "MeV/c", "μm"]

unitsDictionary = dict(zip(allFeatures["any"], featureUnits))



def readRootFile(rootFilePath, features):

    with uproot.open(rootFilePath) as TChain: TTree = TChain["DecayTree"]

    rootFile = TTree.arrays(features, library="pd", cut = "(abs(B_M01-895.55)<100)", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
    rootFile = rootFile.reset_index(drop=True)
    return rootFile

def rootToCSV(rootPath=fileNamesRoot["kpi"], csvPath=fileNamesCSV["kpi"], variables=allFeatures["kpi"] + ["NB0_Kpigamma_sw"]):
    
    LoadedData = readRootFile(rootPath, variables)
    LoadedData.to_csv(csvPath)
    print(f"Data saved to csv file {csvPath}")

def roundUp(data, cutOff=0.5):

    predictions = []

    for prob in data:
        if prob > cutOff: prediction = 1
        else: prediction = 0
        predictions.append(float(prediction))

    return np.array(predictions)

def NNSummaryToSring(NNmodel):
    stringlist = []
    NNmodel.summary(print_fn=lambda x: stringlist.append(x))
    modelSummary = "\n".join(stringlist)
    return modelSummary





if __name__ == "__main__":

    from classifierAnalysis import *

    createBestClassifiers(name="Best", mode="pipi")
    createBestClassifiers(name="Best", mode="kpi")
    createTestClassifiers(name="Test", mode="pipi")
    createTestClassifiers(name="Test", mode="kpi")
    
