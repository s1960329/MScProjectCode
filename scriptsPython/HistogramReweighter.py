
import numpy             as np

from hep_ml import reweight
from misc   import sharedVariables, fullVariables, dataPath
from misc   import readRootFile, rootToCSV


class GBReweight():

    def __init__(self, monteCarloRootFile, sampleRootFile, decayMode = "kpi"):
        self.monteCarloRootFile  = monteCarloRootFile
        self.sampleRootFile      = sampleRootFile

        self.decayMode           = decayMode
        self.trainingVariables   = sharedVariables[:5] + [fullVariables[decayMode][10]]
        self.readData()

    def readData(self):
        self.monteCarloData      = readRootFile(self.monteCarloRootFile, fullVariables[self.decayMode])
        self.monteCarloWeights   = np.ones(len(self.monteCarloData))

        self.sampleData          = readRootFile(self.sampleRootFile, fullVariables["kpi"] + ["NB0_Kpigamma_sw"])
        self.sampleWeights       = np.array(self.sampleData["NB0_Kpigamma_sw"])
        self.sampleData          = self.sampleData.drop(labels=["NB0_Kpigamma_sw"], axis="columns")

        print("Sample and Monte Carlo Data has been loaded...")

    def trainReweighter(self):
        sampleTrainingVariables = sharedVariables[:5] + [fullVariables["kpi"][10]]
        self.reweighter = reweight.GBReweighter(n_estimators=250, learning_rate=0.02, max_depth=4, min_samples_leaf=1000, gb_args={'subsample': 0.4})
        self.reweighter.fit(self.monteCarloData[self.trainingVariables], self.sampleData[sampleTrainingVariables], original_weight=self.monteCarloWeights, target_weight=self.sampleWeights)
        print("Histogram reweighter has been trained...")

    def computeWeights(self):
        fittedWeights            = self.reweighter.predict_weights(self.monteCarloData[self.trainingVariables])
        Norm                     = len(fittedWeights)/sum(fittedWeights)
        self.fittedWeightsNorm   = np.array([Norm*el for el in fittedWeights])
        
        self.monteCarloData["weights"] = self.fittedWeightsNorm
        print("Normalised Weights have been saved...")

    def saveMonteCarloDataToCSV(self, filepath = "data/Raw/SignalData.csv" ):
        self.monteCarloData.to_csv(filepath)
        print(f"Monte Carlo Data saved to csv file {filepath}")

    def executeAll(self,filepath):
        self.trainReweighter()
        self.computeWeights()
        self.saveMonteCarloDataToCSV(filepath)


def createAllSignals():
    pipiMonteCarloPath  =  "../data/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root"
    kpiMonteCarloPath   =  "../data/dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root"
    samplePath          =  "../data/dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root"

    print("- pipi signal \n")
    pipi = GBReweight(monteCarloRootFile = pipiMonteCarloPath, sampleRootFile = samplePath, decayMode = "pipi")
    pipi.executeAll(filepath= "data/pipi/signal.csv")

    print("- kpi signal \n")
    kpi = GBReweight(monteCarloRootFile = kpiMonteCarloPath, sampleRootFile = samplePath, decayMode = "kpi")
    kpi.executeAll(filepath= "data/kpi/signal.csv")


def createAllBackground():
    print("- pipi sideband data \n")
    pipiSidebandData = "../data/dataROOT/pipiG_sideband_2018_rhoMass.root"
    csvpath = "data/pipi/background.csv"
    LoadedData = readRootFile(pipiSidebandData, fullVariables["pipi"])
    LoadedData.to_csv(csvpath)
    print(f"Data saved to csv file {csvpath}")

    print("- kpi sample data \n")
    kpiSidebandData = "../data/dataROOT/kpiG_sideband_2018.root"
    csvpath = "data/kpi/background.csv"
    LoadedData = readRootFile(kpiSidebandData, fullVariables["kpi"])
    LoadedData.to_csv(csvpath)
    print(f"Data saved to csv file {csvpath}")


if __name__ == "__main__":
    createAllBackground()


