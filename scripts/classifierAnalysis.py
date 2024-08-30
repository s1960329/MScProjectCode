from   classifiers       import *
import matplotlib.pyplot as plt

from misc import *


histStyle         = {"bins"     : 50, 
                     "alpha"    : 0.8,
                     "density"  : True,
                     "histtype" : "step"}


def initCreateClassifiers(name = "Best", mode = "pipi"):

    fullName        = f"{name}{mode}"
    inputFeatures   = ["nTracks", "B_P", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]
    FullData        = pd.read_csv(f"data/{mode}/FullData.csv",  index_col=0)

    GlobalParams    = { "name"           : fullName,
                        "inputFeatures"  : inputFeatures,
                        "inputFullData"  : FullData}
    
    return GlobalParams


def createBestClassifiers(name = "Best", mode = "pipi"):

    GlobalParams  = initCreateClassifiers(name, mode)
    
    inputFeatures = GlobalParams["inputFeatures"]
    
    hiddenLayers  =  [Dense(len(inputFeatures),    input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                      Dropout(0.2),
                      Dense(len(inputFeatures)*12, input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                      Dropout(0.2),
                      Dense(len(inputFeatures),    input_shape = (len(inputFeatures)*12,), activation="relu", kernel_initializer="random_normal"),
                      Dropout(0.2),
                      Dense(1,                      input_shape = (len(inputFeatures),),    activation="sigmoid",    kernel_initializer="random_normal")]
    
    print("- Neural Network")
    NN = NeuralNetworkClassifier(hiddenLayers, **GlobalParams)
    NN.createInFull()

    print("- Random Forest")
    RF = ForestClassifier(RandomForestClassifier(n_estimators=1500, max_depth=6, verbose=1), **GlobalParams)
    RF.createInFull()
    
    print("- AdaBoost")
    AD = ForestClassifier(AdaBoostClassifier(n_estimators=500, learning_rate=0.1), **GlobalParams)
    AD.createInFull()

    print("- Gradient Boost")
    GB = ForestClassifier(GradientBoostingClassifier(n_estimators=750, learning_rate=0.01, max_depth=5, verbose=1), **GlobalParams)
    GB.createInFull()

    return [RF,AD,GB,NN]


def createTestClassifiers(name = "Test", mode = "pipi"):

    GlobalParams  = initCreateClassifiers(name, mode)
    inputFeatures = GlobalParams["inputFeatures"]
    
    hiddenLayers  =  [Dense(1, input_shape = (len(inputFeatures),), activation="sigmoid", kernel_initializer="random_normal")]
    
    print("- Neural Network")
    NN = NeuralNetworkClassifier(hiddenLayers, **GlobalParams)
    NN.epochs = 5
    NN.createInFull()

    print("- Random Forest")
    RF = ForestClassifier(RandomForestClassifier(n_estimators=50, max_depth=6, verbose=1), **GlobalParams)
    RF.createInFull()
    
    print("- AdaBoost")
    AD = ForestClassifier(AdaBoostClassifier(n_estimators=50, learning_rate=0.1), **GlobalParams)
    AD.createInFull()

    print("- Gradient Boost")
    GB = ForestClassifier(GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, verbose=1), **GlobalParams)
    GB.createInFull()

    return [RF,AD,GB,NN]


def createVariableClassifiers(name, mode, modelDict):

    GlobalParams  = initCreateClassifiers(name, mode)
 
    print("- Neural Network")
    NN = NeuralNetworkClassifier(modelDict["NN"], **GlobalParams)
    NN.createInFull()

    print("- Random Forest")
    RF = ForestClassifier(modelDict["RF"], **GlobalParams)
    RF.createInFull()
    
    print("- AdaBoost")
    AD = ForestClassifier(modelDict["AD"], **GlobalParams)
    AD.createInFull()

    print("- Gradient Boost")
    GB = ForestClassifier(modelDict["GB"], **GlobalParams)
    GB.createInFull()

    return [RF,AD,GB,NN]


def loadModels(name = "Bestpipi"):
    RF = joblib.load(f"savedModels/{name}/RF{name}.joblib")
    AD = joblib.load(f"savedModels/{name}/AD{name}.joblib")
    GB = joblib.load(f"savedModels/{name}/GB{name}.joblib")
    NN = joblib.load(f"savedModels/{name}/NN{name}.joblib")
    return [RF,AD,GB,NN]


def plotROCcurves(models, modelName = ""):
    
    plt.figure(figsize=plotDim)
    plt.plot([0,1],[0,1], c="black", linestyle=":")
    plt.grid(linestyle="--", alpha=0.3)
    plt.title(f"ROC curves {modelName}", fontsize=plotTitleSize)
    plt.xlabel("False Positive Rate", fontsize=plotAxisSize)
    plt.ylabel("True Postive Rate", fontsize=plotAxisSize)

    for model in models:
        Y_test_prediction = model.predict()
        falsePositiveRate, truePositiveRate, threshold = roc_curve(y_true=model.Y_test, y_score=Y_test_prediction)
        plt.plot(falsePositiveRate, truePositiveRate, color=model.color, label=f"{model.abbreviation}")
    
    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)
    
    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/ROCcurves/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+model.name[2:], dpi=227)
    plt.close()


def plotSeperatedPredictions(model):

    predictionData = list(zip(model.predict(),np.array(model.Y_test), np.array(model.W_test)))
  
    signalPredictions     = [p for (p,t,w) in predictionData if t == 1.0]
    backgroundPredictions = [p for (p,t,w) in predictionData if t == 0.0]

    signalweights         = [w for (p,t,w) in predictionData if t == 1.0]
    backgroundweights     = [w for (p,t,w) in predictionData if t == 0.0]
    
    plt.figure(figsize=plotDim)
    plt.hist( [signalPredictions, backgroundPredictions],
    weights = [signalweights,     backgroundweights],
    label   = ["Signal",          "Background"],
    color   = [model.color,       modelColors["BK"]], 
    **histStyle)

    plt.title(f"{model.name} Prediction Distribution", fontsize=plotTitleSize)
    plt.xlabel("Prediction Probability", fontsize=plotAxisSize)
    plt.ylabel("Distribution", fontsize=plotAxisSize)

    plt.axvline(x = model.cut, c = colors["black"], linestyle=":", label=f"Cut: {model.cut}", alpha=0.5)
    
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3)

    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)

    path = f"imgs/predictionDistributions/{model.name[2:]}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+model.name, dpi=227)
    plt.close()


def plotEfficiencies(model, inputData):

    plt.figure(figsize=plotDim)
    signalData = inputData[inputData["isSignal"] == 1.0]
    backgrData = inputData[inputData["isSignal"] == 0.0]

    sp = model.predict(signalData)
    bp = model.predict(backgrData)

    signalData["predicts"] = sp
    backgrData["predicts"] = bp

    cuts   = np.linspace(0,1,1001)
    sigEff = np.array([len(signalData[ (signalData["predicts"] > cut) ]) / len(signalData) for cut in cuts])
    bacEff = np.array([len(backgrData[ (backgrData["predicts"] > cut) ]) / len(backgrData) for cut in cuts])

    difference = sigEff - bacEff
    EffDict = dict(zip(difference, cuts))
    bestCut = EffDict[max(difference)]

    print("Greatest Difference: ", bestCut)
    
    plt.grid(linestyle="--", alpha=0.3)

    plt.title(f"{model.name} Efficiencies", fontsize=plotTitleSize)
    plt.xlabel("Probability", fontsize=plotAxisSize)
    plt.ylabel("Efficiency", fontsize=plotAxisSize)

    plt.plot(cuts, sigEff,     label="Signal",     color = model.color, alpha = 0.8)
    plt.plot(cuts, bacEff,     label="Background", color = modelColors["BK"], alpha = 0.8)
    plt.axvline(x = bestCut, c ="black", linestyle=":", label=f"Cut: {np.round(bestCut,3)}", alpha=0.5)
    plt.legend()

    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)

    path = f"imgs/efficiencies/{model.name[2:]}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+model.name, dpi=227)
    plt.close()


def plotVariableDistribution(feature, model):

    inputData = model.X_test
    inputData["isSignal"] = model.Y_test
    inputData["weights"]  = model.W_test

    inputData["Prediction"] = roundUp(model.predict(inputData), model.cut)

    SignalData     = inputData[inputData["Prediction"] == 1]
    BackgroundData = inputData[inputData["Prediction"] == 0]

    mean = np.mean(inputData[feature])
    std  = np.std(inputData[feature])

    minRange = min(inputData[feature])
    maxRange = min(max(inputData[feature]), mean + 2*std) 
    
    plt.grid(axis="both", linestyle="dashed", alpha=0.3)
    plt.figure(figsize=plotDim)
    plt.title(f"{model.name} {feature} Distribution", fontsize=plotTitleSize)

    plt.hist( [SignalData[feature], BackgroundData[feature],  inputData[feature]], 
    weights = [SignalData["weights"],BackgroundData["weights"], inputData["weights"]],
    label   = ["Signal","Background","Total"],
    color   = [model.color, modelColors["**"], colors["black"]],
    range   = [minRange,maxRange],  
    bins=50, 
    histtype="step")

    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)

    path = f"imgs/featureDistributions/{model.name}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+feature, dpi=227)
    plt.close()


def plotFigureOfMerit(model, inputData, saveFig = True):

    cuts = np.linspace(0,1,1001)

    (bestCut, FoM) = getFoMandBestCut(cuts, model, inputData)
    plt.figure(figsize=plotDim)
    plt.plot(cuts, FoM, c = model.color, label="Figure of Merit")
    plt.grid(linestyle="--", alpha=0.3)
    plt.title(f"{model.name} Figure of Merit", fontsize = plotTitleSize)
    plt.xlabel("Cut",                          fontsize = plotAxisSize)
    plt.ylabel("Figure of Merit",              fontsize = plotAxisSize)
    
    FoMDict = dict(zip(FoM, cuts))
    bestCut = FoMDict[max(FoM)]
    
    print("Best Cut: ", bestCut)
    plt.axvline(x = bestCut, c = colors["black"], linestyle=":", label=f"Cut: {np.round(bestCut,3)}", alpha=0.5)
    plt.legend()

    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)
    
    path = f"imgs/figuresOfMerit/{model.name[2:]}/"

    if saveFig:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path+model.name, dpi=227)

    plt.close()
    return bestCut


def getConfusionMatrix(model, probcut):
    truth      = model.Y_test
    prediction = roundUp(model.predict(), probcut)
    return confusion_matrix(y_pred=prediction, y_true=truth)


def createBackgroundRemovalDistribution(model, feature, decayMode = "pipi"):

    histStyle =     {"bins"     : 50, 
                     "alpha"    : 0.8,
                     "density"  : True, 
                     "histtype" : "step"}
    plt.figure(figsize=plotDim)
    SampleData     = pd.read_csv(f"data/{decayMode}/SampleDataTest.csv",  index_col=0)
    kpiDict        = dict(zip(uniqueFeatures["kpi" ], uniqueFeatures["any"]))
    SampleData     = SampleData.rename(columns = kpiDict)

    SampleData["isSignal"] = np.ones(len(SampleData))

    SampleData["predictions"] = roundUp(model.predict(SampleData), model.cut)
    SampleDataSansBackground  = SampleData[SampleData["predictions"] == 1.0]

    mean = np.mean(SampleData[feature])
    std  = np.std(SampleData[feature])

    minRange = min(SampleData[feature])
    maxRange = mean + 3*std

    plt.hist( [SampleData[feature],SampleDataSansBackground[feature],SampleData[feature]],
    color =   [colors["black"], model.color, modelColors["**"]],
    label =   ["Unfiltered Data", model.abbreviation + " Cut","sWeights Reweight"],
    weights = [np.ones(len(SampleData)), np.ones(len(SampleDataSansBackground)), SampleData["weights"]],
    range =   [minRange,maxRange],
    **histStyle
    )

    plt.title(f"{model.name} Feature Distributions", fontsize=plotTitleSize)
    plt.xlabel(feature + " " + unitsDictionary[feature], fontsize=plotAxisSize)
    plt.ylabel("Normalised Distribution", fontsize=plotAxisSize)
    plt.grid(axis="both", linestyle="dashed", alpha=0.3)

    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)

    path = f"imgs/sWeightRemovalDistributions/{model.name}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+feature, dpi=227)
    plt.close()


def createTunePlot():
    models = [1,2,3,4,5,6]
    
    logLossTrainRFpipi = [0.33743, 0.33941, 0.27411, 0.27474, 0.24979, 0.24969]
    logLossTestRFpipi  = [0.32702, 0.33134, 0.27193, 0.27243, 0.25394, 0.25402]

    logLossTrainADpipi = [0.65721, 0.67306, 0.65821, 0.67447, 0.61015, 0.62455]
    logLossTestADpipi  = [0.65671, 0.67290, 0.65794, 0.67436, 0.60889, 0.62356]
    
    plt.plot(models, logLossTrainRFpipi, label="Train")
    plt.plot(models, logLossTestRFpipi, label="Test")
    plt.show()

    plt.plot(models, logLossTrainADpipi, label="Train")
    plt.plot(models, logLossTestADpipi, label="Test")
    plt.show()


def createMassDistribution(modelname="NN", name = "Best", mode = "pipi", saveFig=True, removeSignalRegion="default"):

    histStyle        = { "bins"     : 50,
                         "alpha"    : 0.8,
                         "density"  : False,
                         "histtype" : "step"}
    
    modelNames    = ["RF","AD","GB","NN"]
    [RF,AD,GB,NN] = loadModels(name+mode)

    model = dict(zip(modelNames,[RF,AD,GB,NN]))[modelname]

    realData = readRootFile(fileNamesRoot[mode+"pf"],allFeatures[mode]+["B_MM"])
    varDict  = dict(zip(uniqueFeatures[mode],uniqueFeatures["any"]))
    realData = realData.rename(columns = varDict)
    realData["isSignal"] = np.ones(len(realData))
    realData["weights"]  = np.ones(len(realData))

    if removeSignalRegion == "remove" and mode == "pipi":
        realData = realData[ (realData["B_MM"] < 5155) | (realData["B_MM"] > 5493) ]

    realData["prediction"] = model.classify(inputData=realData)
    pipiSignal = realData[realData["prediction"] == 1.0]

    plt.figure(figsize=plotDim)
    plt.grid(linestyle="--", alpha=0.3)
    plt.title(f"{mode} B_MM distribution", fontsize = plotTitleSize )
    plt.xlabel("B_MM $(MeV/c^{2})$", fontsize = plotAxisSize)
    plt.hist([realData["B_MM"], pipiSignal["B_MM"]], 
    label  = [f"Before {modelname} Cut", f"After {modelname} Cut"],
    color  = [colors["black"], modelColors[modelname]],         
             **histStyle)
    
    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)
    
    path = f"imgs/massDistributions/{removeSignalRegion}/{name}{mode}/{modelname}{name}{mode}.png"

    if saveFig:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=227)

    plt.close()


def createAllMassDistibutions():

    removeSignalRegion = ["default","remove"]    

    for mask in removeSignalRegion:

        modelNames    = ["RF","AD","GB","NN"]
        groupNames    = ["Best","Test"]
        modes         = ["pipi","kpi"]

        for modelName in modelNames:
            for groupName in groupNames:
                for mode in modes:
                    createMassDistribution(modelName, groupName, mode, saveFig=True, removeSignalRegion=mask)

if __name__ == "__main__":
    createBestClassifiers()

