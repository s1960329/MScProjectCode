import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PlotCreator():

    def __init__(self) -> None:
        self.variables  = ["nTracks","B_P","B_Cone3_B_ptasy","B_MINIPCHI2","B_FD_OWNPV","gamma_PT","piminus_IP_OWNPV","piminus_PT"]
        self.modes      = ["kpi","kpisw", "pipi"]
        self.hist_style = {"bins" : 100, "density" : True, "alpha" : 1, "histtype" : "step"}
        self.colors     = {"kpi" : "#066bd6", "kpisw"  : "#d60645",  "pipi" : "#427a01", "ratio":"#000000"}
        
        self.loadData()

    def loadData(self):
        self.dataframes  = {}
        path = "/Users/finnjohnonori/Documents/GitHubRepositories/HistogramsLHCbFull/data/"

        # Imports the Kpi monte carlo data
        Kpi_TChain_MC_up     = uproot.open(path + "kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        Kpi_TChain_MC_down   = uproot.open(path + "kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")

        Kpi_TTree_MC_down    = Kpi_TChain_MC_down["DecayTree"]
        Kpi_down_MC_df       = Kpi_TTree_MC_down.arrays(self.variables, library="pd") # type: ignore

        Kpi_TTree_up         = Kpi_TChain_MC_up["DecayTree"]
        Kpi_up_MC_df         = Kpi_TTree_up.arrays(self.variables, library="pd") # type: ignore

        self.dataframes["kpi"]     = pd.concat((Kpi_up_MC_df, Kpi_down_MC_df))

        # Imports the Kpi monte carlo data
        pipi_TChain_MC_up     = uproot.open(path + "pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        pipi_TChain_MC_down   = uproot.open(path + "pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")

        pipi_TTree_MC_down    = pipi_TChain_MC_down["DecayTree"]
        pipi_down_MC_df       = pipi_TTree_MC_down.arrays(self.variables, library="pd") # type: ignore

        pipi_TTree_up         = pipi_TChain_MC_up["DecayTree"]
        pipi_up_MC_df         = pipi_TTree_up.arrays(self.variables, library="pd") # type: ignore

        self.dataframes["pipi"]     = pd.concat((pipi_up_MC_df, pipi_down_MC_df))

        #Imports the Sample Data
        Kpi_TChain_SM        = uproot.open(path + "Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root")
        Kpi_TTree_SM         = Kpi_TChain_SM["DecayTree"]
        self.dataframes["kpisw"]  = Kpi_TTree_SM.arrays(self.variables + ["NB0_Kpigamma_sw"], library="pd", cut = "(abs(B_M01-895.55)<100)") # type: ignore

    def createHistogram(self, variable, decay_mode):
        plt.hist(self.dataframes[decay_mode][variable], color=self.colors[decay_mode], **self.hist_style)
        dist_trained.set_xlim(float("%.2g" % min(hist_total_x)), float("%.2g" % max(hist_total_x)))
        rati_trained.set_ylim(0, max(ratio_data)*1.1)
        plt.show()



if __name__ == "__main__":
    p = PlotCreator()
    p.createHistogram(variable="gamma_PT", decay_mode="pipi")