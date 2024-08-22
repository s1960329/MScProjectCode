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
    GB = ForestClassifier(GradientBoostingClassifier(n_estimators=800, learning_rate=0.012, max_depth=4, verbose=1), **GlobalParams)
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


def plotROCcurves(models):

    plt.plot([0,1],[0,1], c="black", linestyle=":")
    plt.grid(linestyle="--", alpha=0.3)
    plt.title("ROC curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postive Rate")


    for model in models:
        Y_test_prediction = model.predict()
        falsePositiveRate, truePositiveRate, threshold = roc_curve(y_true=model.Y_test, y_score=Y_test_prediction)
        plt.plot(falsePositiveRate, truePositiveRate, color=model.color, label=f"{model.abbreviation}")
    
    plt.legend()
    
    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/ROCcurves/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+model.name[2:], dpi=227)
    plt.close()


def plotSeperatedPredictions(model, bestCut=None):

    predictionData = list(zip(model.predict(),np.array(model.Y_test), np.array(model.W_test)))
  
    signalPredictions     = [p for (p,t,w) in predictionData if t == 1.0]
    backgroundPredictions = [p for (p,t,w) in predictionData if t == 0.0]

    signalweights         = [w for (p,t,w) in predictionData if t == 1.0]
    backgroundweights     = [w for (p,t,w) in predictionData if t == 0.0]

    plt.hist( [signalPredictions, backgroundPredictions],
    weights = [signalweights,     backgroundweights],
    label   = ["Signal",          "Background"],
    color   = [model.color,       modelColors["BK"]], 
    **histStyle)

    plt.title(f"{model.name} Prediction Distribution")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Distribution")

    if bestCut is not None:
        plt.axvline(x = bestCut, c = colors["black"], linestyle=":", label=f"Cut: {bestCut}", alpha=0.5)
    
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3)

    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/predictionDistributions/{model.name[2:]}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+model.name, dpi=227)
    plt.close()


def plotEfficiencies(model, inputData):

    signalData = inputData[inputData["isSignal"] == 1.0]
    backgrData = inputData[inputData["isSignal"] == 0.0]

    sp = model.predict(model.normaliseData(signalData))
    bp = model.predict(model.normaliseData(backgrData))

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

    plt.title(f"{model.name} Efficiencies")
    plt.xlabel("Probability")
    plt.ylabel("Efficiency")

    plt.plot(cuts, sigEff,     label="Signal",     color = model.color, alpha = 0.8)
    plt.plot(cuts, bacEff,     label="Background", color = modelColors["BK"], alpha = 0.8)
    plt.axvline(x = bestCut, c ="black", linestyle=":", label=f"Cut: {np.round(bestCut,3)}", alpha=0.2)
    plt.legend()

    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/efficiencies/{model.name[2:]}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+model.name, dpi=227)
    plt.close()


def plotVariableDistribution(variable, model, inputData = pd.DataFrame(), cut = 0.5):

    if inputData.empty:
        inputData = model.X_test
        inputData["isSignal"] = model.Y_test
        inputData["weights"]  = model.W_test

    X_Data = inputData[model.inputFeatures]
    inputData["Prediction"] = roundUp(model.predict(X_Data), cut)

    SignalData     = inputData[inputData["Prediction"] == 1]
    BackgroundData = inputData[inputData["Prediction"] == 0]

    plt.title(f"{variable} Prediction Distribution")
    plt.hist( [SignalData[variable], BackgroundData[variable],  inputData[variable]], 
    weights = [SignalData["weights"],BackgroundData["weights"], inputData["weights"]],
    label   = ["Signal","Background","Total"], 
    bins=50, 
    histtype="step")
    plt.legend()
    plt.show()


def getFoMandBestCut(cuts, model, inputData):
    
    inputData["prediction"] = model.predict(model.normaliseData(inputData))

    signalData = inputData[inputData["isSignal"] == 1.0]

    FoM = []
    for cut in cuts:
        sE    = len(signalData[ (signalData["prediction"] > cut) ]) / len(signalData)
        Bcomb = len(inputData[  (inputData["prediction"]  > cut) ])
        FoM.append(len(signalData)*sE / np.sqrt(len(signalData)*sE + Bcomb))

        
    FoMDict = dict(zip(FoM, cuts))
    bestCut = FoMDict[max(FoM)]

    return (bestCut, FoM)


def plotFigureOfMerit(model, inputData):

    cuts = np.linspace(0,1,1001)

    (bestCut, FoM) = getFoMandBestCut(cuts, model, inputData)

    plt.plot(cuts, FoM, c = model.color, label="Figure of Merit")
    plt.grid(linestyle="--", alpha=0.3)
    plt.title(f"{model.name} Figure of Merit")
    plt.xlabel("Cut")
    plt.ylabel("Figure of Merit")
    
    FoMDict = dict(zip(FoM, cuts))
    bestCut = FoMDict[max(FoM)]
    
    print("Best Cut: ", bestCut)
    plt.axvline(x = bestCut, c = colors["black"], linestyle=":", label=f"Cut: {np.round(bestCut,3)}", alpha=0.5)
    plt.legend()

    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/figuresOfMerit/{model.name[2:]}/"

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

    SampleData     = pd.read_csv(f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/{decayMode}/SampleDataTest.csv",  index_col=0)
    kpiDict        = dict(zip(uniqueFeatures["kpi" ], uniqueFeatures["any"]))
    SampleData     = SampleData.rename(columns = kpiDict)

    SampleData["predictions"] = roundUp(model.predict(SampleData))
    SampleDataSansBackground  = SampleData[SampleData["predictions"] == 1.0]

    mean = np.mean(SampleData[feature])
    std  = np.std(SampleData[feature])

    minRange = min(SampleData[feature])
    maxRange = min(max(SampleData[feature]), mean + 3*std) 

    plt.hist( [SampleData[feature],SampleDataSansBackground[feature],SampleData[feature]],
    color =   [colors["black"], model.color, modelColors["**"]],
    label =   ["Raw", model.abbreviation,"Sweights"],
    weights = [np.ones(len(SampleData)), np.ones(len(SampleDataSansBackground)), SampleData["weights"]],
    range =   [minRange,maxRange],
    **histStyle
    )

    plt.title(f"{model.name} Feature Distributions")
    plt.xlabel(feature + " " + unitsDictionary[feature])
    plt.ylabel("Normalised Distribution")
    plt.grid(axis="both", linestyle="dashed", alpha=0.3)
    plt.legend()
    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/RemovalDistributions/{model.name}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+feature, dpi=227)
    plt.close()

if __name__ == "__main__":

    [RF,AD,GB,NN] = loadModels(name="Bestkpi")
    for feature in allFeatures["any"] : createBackgroundRemovalDistribution(GB, feature)




    

