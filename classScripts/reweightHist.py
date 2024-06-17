import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
from hep_ml import reweight
from hep_ml.metrics_utils import ks_2samp_weighted
from sklearn.model_selection import train_test_split

class reweightHist():


    def __init__(self) -> None:
        self.columns    = ["gamma_PT","piminus_PT"]#,"piminus_IP_OWNPV","Kplus_IP_OWNPV","Kplus_PT","Kst_892_0_PT","Kst_892_0_IP_OWNPV","B_MINIPCHI2","B_FD_OWNPV","B_Cone3_B_ptasy","nTracks"]
        self.modes      = ["kpi","kpisw", "pipi"]
        self.hist_style = {"bins" : 100, "density" : True, "alpha" : 1, "histtype" : "step"}
        self.colors     = {"kpi" : "#066bd6", "kpisw"  : "#d60645",  "pipi" : "#0000ff", "ratio":"#000000"}
        
        self.loadData()
        self.splitData()


    def loadData(self):
        self.dataframes  = {}
        path = "/Users/finnjohnonori/Documents/GitHubRepositories/HistogramsLHCbFull/data/"

        # Imports the Kpi monte carlo data - Original
        with uproot.open(path + "kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root") as Kpi_TChain_MC_up: # type: ignore
            Kpi_TTree_up         = Kpi_TChain_MC_up["DecayTree"]
            Kpi_up_MC_df         = Kpi_TTree_up.arrays(self.columns, library="pd") # type: ignore

        with uproot.open(path + "kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root") as Kpi_TChain_MC_down: # type: ignore
            Kpi_TTree_MC_down    = Kpi_TChain_MC_down["DecayTree"]
            Kpi_down_MC_df       = Kpi_TTree_MC_down.arrays(self.columns, library="pd") # type: ignore

        self.dataframes["kpi"]   = pd.concat((Kpi_up_MC_df, Kpi_down_MC_df))

        # Import the pipi monte carlo data
        # with uproot.open(path + "pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magup.root") as pipi_TChain_MC_up: # type: ignore
        #     pipi_TTree_MC_up     = pipi_TChain_MC_up["DecayTree"]
        #     pipi_up_MC_df        = pipi_TTree_MC_up.arrays(self.columns, library="pd") # type: ignore

        # with uproot.open(path + "pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magdown.root") as pipi_TChain_MC_down: # type: ignore
        #     pipi_TTree_MC_down   = pipi_TChain_MC_down["DecayTree"]
        #     pipi_down_MC_df      = pipi_TTree_MC_down.arrays(self.columns, library="pd") # type: ignore

        # self.dataframes["pipi"]  = pd.concat((pipi_up_MC_df, pipi_down_MC_df))

        #Imports the Sample Data - Target
        with uproot.open(path + "Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root") as Kpi_TChain_SM: # type: ignore
            Kpi_TTree_SM         = Kpi_TChain_SM["DecayTree"]
        
        self.dataframes["kpisw"] = Kpi_TTree_SM.arrays(self.columns + ["NB0_Kpigamma_sw"], library="pd", cut = "(abs(B_M01-895.55)<100)") # type: ignore
        self.dataframes["kpisw"] = self.dataframes["kpisw"].reset_index(drop=True)

    
    def splitData(self):
        # divide original samples into training ant test parts
        self.kpiMC_train, self.kpiMC_test = train_test_split(self.dataframes["kpi"])
        # divide target samples into training ant test parts
        self.kpiSW_train, self.kpiSW_test = train_test_split(self.dataframes["kpisw"])

        self.kpiMC_weights_train  = np.ones(len(self.kpiMC_train))
        self.kpiMC_weights_test   = np.ones(len(self.kpiMC_test))

        self.kpiSW_weights_train  = self.kpiSW_train["NB0_Kpigamma_sw"]
        self.kpiSW_weights_test   = self.kpiSW_test["NB0_Kpigamma_sw"]
        

    def CreateStandardReweighter(self):
        self.Reweighter = reweight.BinsReweighter(n_bins=20, n_neighs=1.)
        self.Reweighter.fit(self.kpiMC_train, self.kpiSW_train[self.columns], original_weight=self.kpiMC_weights_train , target_weight=self.kpiSW_weights_train)
        print("Standard Reweighter Created")


    def CreateGradientBoostedReweighter(self):
        self.Reweighter = reweight.GBReweighter(n_estimators=250, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4})
        self.Reweighter.fit(self.kpiMC_train, self.kpiSW_train[self.columns])
        print("Gradient Boosted Reweighter Created")


    def CreateDoublePlot(self, variable, rw="st"):

        #Create First Two Plots
        canvas, ((dist_original, dist_trained),(rati_original, rati_trained) ) = plt.subplots(2,2, gridspec_kw={"height_ratios" : [2,1] },figsize=(15, 7))
        canvas.suptitle(f"{variable} reweighting")
        canvas.tight_layout()

        rati_original.grid(axis="x", linestyle="dashed", alpha=0.5)
        dist_original.grid(axis="x", linestyle="dashed", alpha=0.5)
        rati_trained. grid(axis="x", linestyle="dashed", alpha=0.5)
        dist_trained. grid(axis="x", linestyle="dashed", alpha=0.5)
        
        KS_before = ks_2samp_weighted(self.kpiMC_test[variable], self.kpiSW_test[variable], weights1=self.kpiMC_weights_test, weights2=np.ones(len(self.kpiSW_test[variable]), dtype=float))
        dist_original.set_title("Before KS: " + str(np.round(KS_before,4)))
        

        xLowerBound = min(list(self.kpiMC_test[variable]) + list(self.kpiSW_test[variable]))
        xUpperBound = max(list(self.kpiMC_test[variable]) + list(self.kpiSW_test[variable]))

        #Create First Plot
        hist_kpi_y,   hist_kpi_x,   _  = dist_original.hist(self.kpiMC_test[variable], range=(xLowerBound, xUpperBound), color=self.colors["kpi"  ], label="kpi MC",     **self.hist_style)
        hist_kpisw_y, hist_kpisw_x, _  = dist_original.hist(self.kpiSW_test[variable], range=(xLowerBound, xUpperBound), color=self.colors["kpisw"], label="Kpi Sample", **self.hist_style)
    
        rati_original.set_yscale("log")

        ratio_data = np.nan_to_num( (hist_kpisw_y / hist_kpi_y), nan=-1, posinf=-1, neginf=-1)
        rati_original.scatter(hist_kpisw_x[:-1], ratio_data, s=5, c=self.colors["ratio"])

        FittedWeights = self.Reweighter.predict_weights(self.kpiMC_test)

        KS_After = ks_2samp_weighted(self.kpiMC_test[variable], self.kpiSW_test[variable], weights1=FittedWeights, weights2=np.ones(len(self.kpiSW_test[variable]), dtype=float))
        dist_trained.set_title("After KS: " + str(np.round(KS_After,4)))

        #Create Second Plot
        hist_kpi_y,   hist_kpi_x,   _  = dist_trained.hist(self.kpiMC_test[variable], range=(xLowerBound, xUpperBound), color=self.colors["kpi"  ], weights=FittedWeights, label="kpi MC",  **self.hist_style)
        hist_kpisw_y, hist_kpisw_x, _  = dist_trained.hist(self.kpiSW_test[variable], range=(xLowerBound, xUpperBound), color=self.colors["kpisw"], label="Kpi Sample", **self.hist_style)

        dist_trained.legend()

        ratio_data = np.nan_to_num( (hist_kpisw_y / hist_kpi_y), nan=-1, posinf=-1, neginf=-1)

        rati_trained.scatter(hist_kpisw_x[:-1],ratio_data, s=5, c=self.colors["ratio"])
        rati_trained.grid(axis="x", linestyle="dashed", alpha=0.7)
        dist_trained.grid(axis="x", linestyle="dashed", alpha=0.7)

        rati_trained.set_yscale("log")

        rati_original.set_xlim(xLowerBound, xUpperBound)
        dist_original.set_xlim(xLowerBound, xUpperBound)
        rati_trained.set_xlim(xLowerBound, xUpperBound)
        dist_trained.set_xlim(xLowerBound, xUpperBound)

        #Display the final images
        plt.savefig(f"{variable}.png")

    def createAllDoublePlots(self):
        self.CreateGradientBoostedReweighter()
        for variable in self.columns:
            self.CreateDoublePlot(variable)


if __name__ == "__main__":
    p = reweightHist()
    p.createAllDoublePlots()
    


    