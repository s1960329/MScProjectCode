
import numpy                 as np
from sklearn.model_selection import train_test_split
from misc import *
import matplotlib.pyplot as plt
from hep_ml import reweight
from misc   import sharedFeatures, allFeatures, dataPath
from misc   import readRootFile, rootToCSV


class GBReweight():

    def __call__(self):
        return f"(n_estimators = {self.n_estimators}, learning_rate = {self.learning_rate}, max_depth = {self.max_depth}, min_samples_leaf = {self.min_samples_leaf}, subsample = {self.subsample}, trainingFeatures = {self.trainingFeatures}, decayMode = {self.decayMode})"  
        
    def __init__(self, monteCarloRootFile, sampleRootFile, n_estimators=320, learning_rate=0.025, max_depth=6,  min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5] + [allFeatures["any"][8], allFeatures["any"][10]], decayMode = "kpi"):
        
        self.n_estimators        = n_estimators
        self.learning_rate       = learning_rate
        self.max_depth           = max_depth
        self.min_samples_leaf    = min_samples_leaf
        self.subsample           = subsample 
        
        self.monteCarloRootFile  = monteCarloRootFile
        self.sampleRootFile      = sampleRootFile

        self.decayMode           = decayMode
        self.trainingFeatures    = trainingFeatures 
        self.readData()

    def readData(self):
        self.monteCarloData      = readRootFile(self.monteCarloRootFile, allFeatures[self.decayMode])
        self.monteCarloWeights   = np.ones(len(self.monteCarloData))

        self.sampleData, self.sampleData_test  = train_test_split(readRootFile(self.sampleRootFile, allFeatures["kpi"] + ["NB0_Kpigamma_sw"]), test_size = 0.1, random_state=23)
        self.sampleWeights       = np.array(self.sampleData["NB0_Kpigamma_sw"])
        self.sampleData          = self.sampleData.drop(labels=["NB0_Kpigamma_sw"], axis="columns")

        VarDict    = dict(zip(uniqueFeatures[self.decayMode],uniqueFeatures["any"]))
        KpiDict    = dict(zip(uniqueFeatures["kpi"],uniqueFeatures["any"]))    
        
        self.monteCarloData = self.monteCarloData.rename(columns = VarDict)
        self.sampleData     = self.sampleData.rename(columns = KpiDict)

        print("Sample and Monte Carlo Data has been loaded...")

    def trainReweighter(self):
        self.reweighter = reweight.GBReweighter(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, gb_args={'subsample': self.subsample})
        self.reweighter.fit(self.monteCarloData[self.trainingFeatures], self.sampleData[self.trainingFeatures], original_weight=self.monteCarloWeights, target_weight=self.sampleWeights)
        print("Histogram reweighter has been trained...")

    def computeWeights(self):
        fittedWeights            = self.reweighter.predict_weights(self.monteCarloData[self.trainingFeatures])
        Norm                     = len(fittedWeights)/sum(fittedWeights)
        self.fittedWeightsNorm   = np.array([Norm*el for el in fittedWeights])
        
        self.monteCarloData["weights"] = self.fittedWeightsNorm
        print("Normalised Weights have been saved...")

    def createHist(self, feature):

        histStyle     = {"bins"     : 50, 
                         "alpha"    : 0.8,
                         "density"  : True,
                         "histtype" : "step"}
        
        self.histY, self.histX, _  = plt.hist(  [self.sampleData[feature], self.monteCarloData[feature],   self.monteCarloData[feature]], 
                                     weights =  [self.sampleWeights, self.monteCarloData["weights"], np.ones(len(self.monteCarloData))],
                                     **histStyle)
        

    def saveMonteCarloDataToCSV(self, filepath = "data/Raw/SignalData.csv" ):
        renameDict = dict(zip(allFeatures[self.decayMode], allFeatures["any"]))
        self.monteCarloData = self.monteCarloData.rename(columns=renameDict)
        self.monteCarloData.to_csv(filepath)
        print(f"Monte Carlo Data saved to csv file {filepath}")

    def saveTestData(self):
        SamplecsvPath     = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/{self.decayMode}/SampleDataTest.csv"
        renameDict = dict(zip(allFeatures["kpi"] + ["NB0_Kpigamma_sw"], allFeatures["any"] + ["weights"]))
        self.sampleData_test = self.sampleData_test.rename(columns=renameDict)
        self.sampleData_test.to_csv(SamplecsvPath)

    def executeAll(self,filepath):
        self.trainReweighter()
        self.computeWeights()
        self.saveMonteCarloDataToCSV(filepath)
        self.saveTestData()

def createAllSignals():
    pipiMonteCarloPath  =  "../data/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root"
    kpiMonteCarloPath   =  "../data/dataROOT/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel.root"
    samplePath          =  "../data/dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root"

    print("- pipi signal \n")
    pipi = GBReweight(monteCarloRootFile = pipiMonteCarloPath, sampleRootFile = samplePath, decayMode = "pipi")
    pipi.executeAll(filepath= "data/pipi/SignalData.csv")

    print("- kpi signal \n")
    kpi = GBReweight(monteCarloRootFile = kpiMonteCarloPath, sampleRootFile = samplePath, decayMode = "kpi")
    kpi.executeAll(filepath= "data/kpi/SignalData.csv")

def createAllBackground():
    print("- pipi sideband data \n")
    pipiSidebandData = "../data/dataROOT/pipiG_sideband_2018_rhoMass.root"
    csvpath = "data/pipi/BackgroundData.csv"
    LoadedData = readRootFile(pipiSidebandData, allFeatures["pipi"])
    renameDict = dict(zip(allFeatures["pipi"], allFeatures["any"]))
    LoadedData = LoadedData.rename(columns=renameDict)
    LoadedData.to_csv(csvpath)
    print(f"Data saved to csv file {csvpath}")

    print("- kpi sample data \n")
    kpiSidebandData = "../data/dataROOT/kpiG_sideband_2018.root"
    csvpath = "data/kpi/BackgroundData.csv"
    LoadedData = readRootFile(kpiSidebandData, allFeatures["kpi"])
    renameDict = dict(zip(allFeatures["kpi"], allFeatures["any"]))
    LoadedData = LoadedData.rename(columns=renameDict)
    LoadedData.to_csv(csvpath)
    print(f"Data saved to csv file {csvpath}")


if __name__ == "__main__":
    createAllBackground()


