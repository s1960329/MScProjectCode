from histogramReweighter import *
from formatData          import *
from evaluateReweighter  import *
from classifierAnalysis  import *
from misc                import *


def bestClassifiersOnly():
    ModelsBestpipi = loadModels(name="Bestpipi")
    pipiTestData = pd.read_csv("/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/kpi/TestData.csv",  index_col=0)
    plotROCcurves(ModelsBestpipi)

    for model in ModelsBestpipi :  
        plotSeperatedPredictions(model)
        plotEfficiencies(model,  inputData=pipiTestData)
        plotFigureOfMerit(model, inputData=pipiTestData)


def fullProject():

    print(" --- Reweighting Data")
    createSampleData()
    createAllSignals()
    createAllBackground()

    print(" --- Plotting Reweighted Histograms")
    for feature in allFeatures["any"]  : createReweightedHistogram(feature, decayMode="pipi")
    for feature in allFeatures["any"]  : createReweightedHistogram(feature, decayMode="kpi")
    
    print(" --- Evaluating Reweighter")
    evaluateReweighter()

    print(" --- Formatting Data")
    combineSignalAndBackground(mode = "kpi")
    combineSignalAndBackground(mode = "pipi")

    createTestAndTrainData(mode = "kpi")
    createTestAndTrainData(mode = "pipi")

    createEvenTestAndTrainData(mode = "kpi")
    createEvenTestAndTrainData(mode = "pipi")

    pipiTestData = pd.read_csv("/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/kpi/TestData.csv",  index_col=0)
    kpiTestData  = pd.read_csv("/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/pipi/TestData.csv", index_col=0)

    print(" --- Creating Classifiers")
    createBestClassifiers(name="Best", mode="pipi")
    createBestClassifiers(name="Best", mode="kpi")
    createTestClassifiers(name="Test", mode="pipi")
    createTestClassifiers(name="Test", mode="kpi")

    print(" --- Load Models")
    ModelsBestpipi = loadModels(name="Bestpipi")
    ModelsTestpipi = loadModels(name="Testpipi")
    ModelsBestkpi  = loadModels(name="Bestkpi")
    ModelsTestkpi  = loadModels(name="Testkpi")

    print(" --- Creating Prediction Distributions")
    for model in ModelsBestpipi :  plotSeperatedPredictions(model)
    for model in ModelsTestpipi :  plotSeperatedPredictions(model)
    for model in ModelsBestkpi  :  plotSeperatedPredictions(model)
    for model in ModelsTestkpi  :  plotSeperatedPredictions(model)

    print(" --- Creating Efficiency Curves")
    for model in ModelsBestpipi :  plotEfficiencies(model, inputData=pipiTestData)
    for model in ModelsTestpipi :  plotEfficiencies(model, inputData=pipiTestData)
    for model in ModelsBestkpi  :  plotEfficiencies(model, inputData=kpiTestData)
    for model in ModelsTestkpi  :  plotEfficiencies(model, inputData=kpiTestData)

    print(" --- Creating Figures of Merit")
    for model in ModelsBestpipi :  plotFigureOfMerit(model, inputData=pipiTestData)
    for model in ModelsTestpipi :  plotFigureOfMerit(model, inputData=pipiTestData)
    for model in ModelsBestkpi  :  plotFigureOfMerit(model, inputData=kpiTestData)
    for model in ModelsTestkpi  :  plotFigureOfMerit(model, inputData=kpiTestData)

    print(" --- Creating ROC curves")
    plotROCcurves(ModelsBestpipi)
    plotROCcurves(ModelsTestpipi)
    plotROCcurves(ModelsBestkpi)
    plotROCcurves(ModelsTestkpi)


if __name__ == "__main__":
    fullProject()