
import numpy as np

from hep_ml import reweight
from misc   import sharedVariables, fullVariables, fileNamesRoot, fileNamesCSV
from misc   import readRootFile, rootToCSV


class GBReweight():

    def __init__(self, decayMode = "kpi"):
        self.decayMode           = decayMode
        self.trainingVariables   = sharedVariables[:5] + [fullVariables[decayMode][10]]

    def executeAll(self):
        self.readData()
        self.trainReweighter()
        self.computeWeights()
        self.saveMonteCarloDataToCSV()

    def readData(self):
        monteCarloPath           = fileNamesRoot[self.decayMode]
        samplePath               = fileNamesRoot["sm"]

        self.monteCarloData      = readRootFile(monteCarloPath, fullVariables[self.decayMode])
        self.monteCarloWeights   = np.ones(len(self.monteCarloData))

        self.sampleData          = readRootFile(samplePath, fullVariables["kpi"] + ["NB0_Kpigamma_sw"])
        self.sampleWeights       = np.array(self.sampleData["NB0_Kpigamma_sw"])
        self.sampleData          = self.sampleData.drop(labels=["NB0_Kpigamma_sw"], axis="columns")
        print("Sample and Monte Carlo Data has been loaded...")

    def trainReweighter(self):
        sampleTrainingVariables = sharedVariables[:5] + [fullVariables["kpi"][10]]
        self.reweighter = reweight.GBReweighter(n_estimators=250, learning_rate=0.02, max_depth=4, min_samples_leaf=1000, gb_args={'subsample': 0.4})
        self.reweighter.fit(self.monteCarloData[self.trainingVariables], self.sampleData[sampleTrainingVariables], original_weight=self.monteCarloWeights, target_weight=self.sampleWeights)
        print("Histogram reweighter has been trained...")

    def computeWeights(self):
        fittedWeights = self.reweighter.predict_weights(self.monteCarloData[self.trainingVariables])
        Norm = len(fittedWeights)/sum(fittedWeights)
        self.fittedWeightsNormalised   = np.array([Norm*el for el in fittedWeights])
        self.monteCarloData["weights"] = self.fittedWeightsNormalised
        print("Normalised Weights have been saved...")

    def saveMonteCarloDataToCSV(self):
        self.monteCarloData.to_csv(fileNamesCSV[self.decayMode])
        print(f"Monte Carlo Data saved to csv file {fileNamesCSV[self.decayMode]}")


if __name__ == "__main__":

    print()

    print("- kpi sideband data")
    rootToCSV(decayMode = "kpisb", variables = fullVariables["kpi"])
    print()

    print("- kpi sample data")
    rootToCSV()
    print()

    print("- kpi monte carlo data")
    kpi  = GBReweight(decayMode="kpi")
    kpi.executeAll()
    print("kpi data has been reweighted")
    print()

    print("- pipi monte carlo data")
    pipi = GBReweight(decayMode="pipi")
    pipi.executeAll()
    print("pipi data has been reweighted")
    print()



