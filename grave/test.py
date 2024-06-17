import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
from hep_ml import reweight
from hep_ml.metrics_utils import ks_2samp_weighted
from sklearn.model_selection import train_test_split

# - Shared Variables
# log(B_MINIPCHI2)
# log(B_FD_OWNPV)
# B_Cone3_B_ptasy
# gamma_PT
# log(piminus_IP_OWNPV)
# piminus_PT

# - K variables
# log(Kplus_IP_OWNPV)
# Kplus_PT
# log(Kst_892_0_PT)
# log(Kst_892_0_IP_OWNPV)

# - 
# log(piplus_IP_OWNPV)
# piplus_PT
# log(rho_770_0_PT)
# log(rho_770_0_IP_OWNPV)

# B_Cone3_B_ptasy

columns    = ["nTracks","B_P","B_Cone3_B_ptasy","B_MINIPCHI2","B_FD_OWNPV","gamma_PT","piminus_IP_OWNPV","piminus_PT"]
modes      = ["kpi","kpisw", "pipi"]
dataframe  = {}
hist_style = {"bins": 100,       "density": True,       "alpha": 0.7}
hist_color = {"kpi" : "#ff0000", "kpisw"  : "#00ff00",  "pipi" : "#0000ff"}

# Imports the Kpi monte carlo data
Kpi_TChain_MC_up     = uproot.open("../data/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
Kpi_TChain_MC_down   = uproot.open("../data/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")

Kpi_TTree_MC_down    = Kpi_TChain_MC_down["DecayTree"]
Kpi_down_MC_df       = Kpi_TTree_MC_down.arrays(columns, library="pd") # type: ignore

Kpi_TTree_up         = Kpi_TChain_MC_up["DecayTree"]
Kpi_up_MC_df         = Kpi_TTree_up.arrays(columns, library="pd") # type: ignore

dataframe["kpi"]     = pd.concat((Kpi_up_MC_df, Kpi_down_MC_df))

# Imports the Kpi monte carlo data
pipi_TChain_MC_up     = uproot.open("../data/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
pipi_TChain_MC_down   = uproot.open("../data/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")

pipi_TTree_MC_down    = pipi_TChain_MC_down["DecayTree"]
pipi_down_MC_df       = pipi_TTree_MC_down.arrays(columns, library="pd") # type: ignore

pipi_TTree_up         = pipi_TChain_MC_up["DecayTree"]
pipi_up_MC_df         = pipi_TTree_up.arrays(columns, library="pd") # type: ignore

dataframe["pipi"]     = pd.concat((pipi_up_MC_df, pipi_down_MC_df))

#Imports the Sample Data
Kpi_TChain_SM        = uproot.open("../data/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root")
Kpi_TTree_SM         = Kpi_TChain_SM["DecayTree"]
dataframe["kpisw"]   = Kpi_TTree_SM.arrays(columns + ["NB0_Kpigamma_sw"], library="pd", cut = "(abs(B_M01-895.55)<100)") # type: ignore



original_weights = np.ones(len(dataframe["kpisw"]))
target_weights   = np.array(dataframe["kpisw"]["NB0_Kpigamma_sw"])

original_train, original_test = train_test_split(dataframe["kpi"])
target_train, target_test = train_test_split(dataframe["kpisw"])

original_weights_train = np.ones(len(original_train))
original_weights_test = np.ones(len(original_test))

print(f"Original train shape: {original_train.shape}")
print(f"Target train shape: {target_train[columns].shape}")
print(f"Original weights train shape: {original_weights_train.shape}")
print(f"Original weights test shape: {original_weights_test.shape}")

def plotHistograms():

    for variable in columns:

        plt.hist(dataframe["kpi"  ][variable], color=hist_color["kpi"  ], label="kpi Sample", **hist_style)
        plt.hist(dataframe["kpisw"][variable], color=hist_color["kpisw"], label="kpi MC",     **hist_style)
        plt.legend()
        plt.savefig(f"../data/{variable}_kpi.png")
        plt.close()

        plt.hist(dataframe["kpi"  ][variable], color=hist_color["kpi"  ], label= "kpi MC" , **hist_style)
        plt.hist(dataframe["pipi" ][variable], color=hist_color["pipi" ], label= "pipi MC", **hist_style)
        plt.legend()
        plt.savefig(f"../data/{variable}_pipi.png")
        plt.close()

