
from hep_ml import reweight

import numpy as np
import pandas as pd
import uproot

dataPath        = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/data/"
imagePath       = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/imgs/"

sharedVariables = ["nTracks", "B_P", "gamma_PT", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV"]

uniqueVariables = {"kpi" : ["Kst_892_0_PT",        "Kst_892_0_IP_OWNPV",        "Kplus_PT",       "Kplus_IP_OWNPV" ],
                   "pipi": ["rho_770_0_PT",        "rho_770_0_IP_OWNPV",        "piplus_PT",      "piplus_IP_OWNPV"],
                   "any" : ["daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_OWNPV"]}

fullVariables = {"kpi"  : sharedVariables + uniqueVariables["kpi"],
                 "pipi" : sharedVariables + uniqueVariables["pipi"],
                 "any"  : sharedVariables + uniqueVariables["any"]}

fileNamesRoot = {"kpi"  : f"{dataPath}dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root",
                 "kpisb": f"{dataPath}dataROOT/kpiG_sideband_2018.root",
                 "pipi" : f"{dataPath}dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root",
                 "sm"   : f"{dataPath}dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root"}

fileNamesCSV  = {"kpi"  : f"{dataPath}dataCSV/kpi_montecarlo_reweighted.csv",
                 "pipi" : f"{dataPath}dataCSV/pipi_montecarlo_reweighted.csv",
                 "kpisb": f"{dataPath}dataCSV/kpi_sideband.csv",
                 "sm"   : f"{dataPath}dataCSV/kpi_sample.csv"}



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


def mergeDataframes():
    pass

