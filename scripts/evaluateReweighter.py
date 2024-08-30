
from misc import *
from histogramReweighter import *

from sklearn.model_selection    import KFold, train_test_split
from classifiers import *
from classifierAnalysis import *
from tensorflow.keras.models    import Sequential, clone_model                     # type: ignore
from tensorflow.keras.layers    import Input, Dense, Dropout, BatchNormalization   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping                               # type: ignore
from scipy.stats                import chisquare, wasserstein_distance
from pyemd import emd

def createStandardDistributions(feature, decayMode, weights=False):

    MonteCarloData = pd.read_csv(fileNamesCSV[decayMode], index_col=0)
    SampleData     = pd.read_csv(fileNamesCSV["sm"],   index_col=0)

    kpiDict  = dict(zip(uniqueFeatures["kpi" ],uniqueFeatures["any"]))
    customDict = dict(zip(uniqueFeatures[decayMode],uniqueFeatures["any"]))

    SampleData     = SampleData.rename(    columns = kpiDict )
    MonteCarloData = MonteCarloData.rename(columns = customDict)

    mean = np.mean(SampleData[feature])
    std  = np.std(SampleData[feature])

    minRange = min(SampleData[feature])
    maxRange = mean + 3*std

    if not weights:
        MClabel   = "Monte Carlo"
        MCcolor   = colors["blue"]
        MCweights = np.ones(len(MonteCarloData))
        prefix    = ""
    else:
        MClabel   = "Weighted Monte Carlo"
        MCcolor   = colors["green"]
        MCweights = MonteCarloData["weights"]
        prefix    = "Weighted"

    plt.figure(figsize=plotDim)
    plt.hist(  [SampleData[feature], MonteCarloData[feature]],
    weights =  [SampleData["NB0_Kpigamma_sw"], MCweights],
    label   =  ["LHCb Signal", MClabel],
    color   =  [colors["red"],  MCcolor],
    range   =  [minRange, maxRange],
    **histStyle
    )

    plt.title(f"Feature Distributions {decayMode}", fontsize=plotTitleSize)
    plt.xlabel(feature + " " + unitsDictionary[feature], fontsize=plotAxisSize)
    plt.ylabel("Normalised Distribution", fontsize=plotAxisSize)
    plt.grid(axis="both", linestyle="dashed", alpha=0.3)
    plt.legend()

    plt.yticks(fontsize=plotLegendSize)
    plt.xticks(fontsize=plotLegendSize)
    plt.legend(fontsize=plotLegendSize)

    path = f"imgs/standardFeatureDistributions/{decayMode}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+feature+prefix, dpi=227)
    plt.close()

def createAllStandardDistributions():
    for feature in allFeatures["any"] : 
        createStandardDistributions(feature, decayMode="pipi", weights=False)
        createStandardDistributions(feature, decayMode="pipi", weights=True)
        createStandardDistributions(feature, decayMode="kpi",  weights=False)
        createStandardDistributions(feature, decayMode="kpi",  weights=True)

def createReweightedHistogram(feature, decayMode):

    MonteCarloData = pd.read_csv(fileNamesCSV[decayMode], index_col=0)
    SampleData     = pd.read_csv(fileNamesCSV["sm"],   index_col=0)

    kpiDict  = dict(zip(uniqueFeatures["kpi" ],uniqueFeatures["any"]))
    customDict = dict(zip(uniqueFeatures[decayMode],uniqueFeatures["any"]))

    SampleData     = SampleData.rename(    columns = kpiDict )
    MonteCarloData = MonteCarloData.rename(columns = customDict)


    canvas, (disti,ratio) = plt.subplots(2,1, gridspec_kw={"height_ratios" : [2,1] },figsize=(8, 7))
    canvas.suptitle(f"{feature} Distribution {unitsDictionary[feature]}")


    histY, histX, _  = disti.hist([SampleData[feature],          MonteCarloData[feature],  MonteCarloData[feature]], 
                       weights =  [SampleData["NB0_Kpigamma_sw"],MonteCarloData["weights"], np.ones(len(MonteCarloData))],
                       label   =  ["Sample",                     "Weighted Monte Carlo",    "Monte Carlo"],
                       color   =  [colors["red"],                colors["green"],           colors["blue"]],
                       **histStyle)

    histBars = dict( zip(["sm","wmc", "mc"],histY) )

    ratioDataMC  = ((histBars["sm"] - histBars["mc" ])**2) / histBars["sm"]
    ratioDataMCW = ((histBars["sm"] - histBars["wmc"])**2) / histBars["sm"]
    ratioDataMC[    (histBars["sm"] < 0) | (histBars["mc" ] < 0)] = np.nan
    ratioDataMCW[   (histBars["sm"] < 0) | (histBars["wmc"] < 0)] = np.nan
    
    ratio.plot(histX[:-1], ratioDataMC,  c=colors["blue"],  alpha=0.8, linewidth=0.7)
    ratio.plot(histX[:-1], ratioDataMCW, c=colors["green"], alpha=0.8, linewidth=0.7)
    ratio.set_title(f"Chi Squared Difference from Sample, Total Change: {float("%.3g" % sum(np.nan_to_num(ratioDataMCW - ratioDataMC)))}")
    ratio.set_xlim(min(histX), max(histX))
    ratio.grid(axis="both", linestyle="dashed", alpha=0.3)

    disti.set_xlim(min(histX), max(histX))
    disti.xaxis.set_tick_params(which = "both",labelbottom=False, bottom=False)
    disti.grid(axis="both", linestyle="dashed", alpha=0.3)
    disti.legend()

    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/reweightedDistributions/{decayMode}/"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+feature, dpi=227)
    plt.close()

def evaluateReweighter(decayMode = "pipi"):

    inputFeatures = allFeatures["any"]

    SampleData     = pd.read_csv("data/SampleData.csv",      index_col=0)
    MonteCarloData = pd.read_csv(f"data/{decayMode}/FullData.csv",   index_col=0)
    MonteCarloData = MonteCarloData[MonteCarloData["isSignal"] == 1.0]
    MonteCarloData = MonteCarloData.drop(columns=["isSignal"])

    SampleData["isSignal"]     = np.ones(len(SampleData))
    MonteCarloData["isSignal"] = np.zeros(len(MonteCarloData))

    sampleSize = min(len(SampleData), len(MonteCarloData))

    SampleData = SampleData.sample(n = sampleSize)
    MonteCarloData = MonteCarloData.sample(n = sampleSize)

    FullData = pd.concat( (SampleData, MonteCarloData))
    FullData = FullData.sample(frac=1)

    print("- with weights")
    GBwithWeights = ForestClassifier(GradientBoostingClassifier(n_estimators=2500, learning_rate=0.005, max_depth=6, verbose=1), "EvaluateRewighterTest", inputFeatures, FullData)
    GBwithWeights.abbreviation = "Weighted"
    GBwithWeights.color        = colors["purple"]
    GBwithWeights.createInFull()

    print("- without weights")
    GBwithoutWeights = ForestClassifier(GradientBoostingClassifier(n_estimators=2500, learning_rate=0.005, max_depth=6, verbose=1), "EvaluateRewighterControl", inputFeatures, FullData)
    GBwithoutWeights.W_train = np.ones(len(GBwithoutWeights.W_train))
    GBwithoutWeights.W_test  = np.ones(len(GBwithoutWeights.W_test))
    GBwithoutWeights.abbreviation = "Unweighted"
    GBwithoutWeights.color        = colors["black"]
    GBwithoutWeights.createInFull()

    plotROCcurves([GBwithWeights,GBwithoutWeights], modelName=decayMode)

def findChiSquaredDifferenceInWeights(model, feature):

    model.createHist(feature)
    histogramDict = dict(zip(["SM","WMC","MC"], model.histY))
    
    WassBefore = wasserstein_distance(histogramDict["SM"], histogramDict["MC"])
    WassAfter  = wasserstein_distance(histogramDict["SM"], histogramDict["WMC"])

    return (WassAfter - WassBefore)

def findTotalWassDiff(n_estimators=40, learning_rate=0.2, max_depth=3, min_samples_leaf=200, subsample=0.4, trainingFeatures=sharedFeatures[:5] + [allFeatures["any"][10]], decayMode = "pipi"):
    
    monteCarloRootFile  =  fileNamesRoot[decayMode]
    sampleRootFile      =  fileNamesRoot["sm"]

    print(f"- {decayMode} signal \n")
    rw = GBReweight(monteCarloRootFile, sampleRootFile, n_estimators, learning_rate, max_depth, min_samples_leaf, subsample, trainingFeatures, decayMode)
    rw.trainReweighter()
    rw.computeWeights()

    total = 0
    for feature in allFeatures["any"]:
        total += findChiSquaredDifferenceInWeights(rw, feature)

    return rw, total

def getAllWassTests(decayMode = "pipi"):

    print("1")
    rw1, total1   = findTotalWassDiff(n_estimators=40,  learning_rate=0.2,   max_depth=3,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("2")
    rw2, total2   = findTotalWassDiff(n_estimators=80,  learning_rate=0.1,   max_depth=3,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("3")
    rw3, total3   = findTotalWassDiff(n_estimators=160, learning_rate=0.05,  max_depth=3,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("4")
    rw4, total4   = findTotalWassDiff(n_estimators=320, learning_rate=0.025, max_depth=3,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("5")
    rw5, total5   = findTotalWassDiff(n_estimators=40,  learning_rate=0.2,   max_depth=6,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("6")
    rw6, total6   = findTotalWassDiff(n_estimators=80,  learning_rate=0.1,   max_depth=6,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("7")
    rw7, total7   = findTotalWassDiff(n_estimators=160, learning_rate=0.05,  max_depth=6,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("8")
    rw8, total8   = findTotalWassDiff(n_estimators=320, learning_rate=0.025, max_depth=6,  min_samples_leaf=200,  subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("9")
    rw9, total9   = findTotalWassDiff(n_estimators=40,  learning_rate=0.2,   max_depth=6,  min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("10")
    rw10, total10 = findTotalWassDiff(n_estimators=80,  learning_rate=0.1,   max_depth=6,  min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("11")
    rw11, total11 = findTotalWassDiff(n_estimators=160, learning_rate=0.05,  max_depth=6,  min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("12")
    rw12, total12 = findTotalWassDiff(n_estimators=320, learning_rate=0.025, max_depth=6,  min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("13")
    rw13, total13 = findTotalWassDiff(n_estimators=40,  learning_rate=0.2,   max_depth=10, min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("14")
    rw14, total14 = findTotalWassDiff(n_estimators=80,  learning_rate=0.1,   max_depth=10, min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("15")
    rw15, total15 = findTotalWassDiff(n_estimators=160, learning_rate=0.05,  max_depth=10, min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)
    print("16")
    rw16, total16 = findTotalWassDiff(n_estimators=320, learning_rate=0.025, max_depth=10, min_samples_leaf=1000, subsample=0.4, trainingFeatures=sharedFeatures[:5], decayMode=decayMode)

    summaryString = "\n"

    summaryString += str(np.round(total1,5))  + f"(n_estimators = {rw1.n_estimators},  learning_rate = {rw1.learning_rate},  max_depth = {rw1.max_depth},  min_samples_leaf = {rw1.min_samples_leaf},  decayMode = {rw1.decayMode})\n"
    summaryString += str(np.round(total2,5))  + f"(n_estimators = {rw2.n_estimators},  learning_rate = {rw2.learning_rate},  max_depth = {rw2.max_depth},  min_samples_leaf = {rw2.min_samples_leaf},  decayMode = {rw2.decayMode})\n"
    summaryString += str(np.round(total3,5))  + f"(n_estimators = {rw3.n_estimators},  learning_rate = {rw3.learning_rate},  max_depth = {rw3.max_depth},  min_samples_leaf = {rw3.min_samples_leaf},  decayMode = {rw3.decayMode})\n"
    summaryString += str(np.round(total4,5))  + f"(n_estimators = {rw4.n_estimators},  learning_rate = {rw4.learning_rate},  max_depth = {rw4.max_depth},  min_samples_leaf = {rw4.min_samples_leaf},  decayMode = {rw4.decayMode})\n"
    summaryString += str(np.round(total5,5))  + f"(n_estimators = {rw5.n_estimators},  learning_rate = {rw5.learning_rate},  max_depth = {rw5.max_depth},  min_samples_leaf = {rw5.min_samples_leaf},  decayMode = {rw5.decayMode})\n"
    summaryString += str(np.round(total6,5))  + f"(n_estimators = {rw6.n_estimators},  learning_rate = {rw6.learning_rate},  max_depth = {rw6.max_depth},  min_samples_leaf = {rw6.min_samples_leaf},  decayMode = {rw6.decayMode})\n"
    summaryString += str(np.round(total7,5))  + f"(n_estimators = {rw7.n_estimators},  learning_rate = {rw7.learning_rate},  max_depth = {rw7.max_depth},  min_samples_leaf = {rw7.min_samples_leaf},  decayMode = {rw7.decayMode})\n"
    summaryString += str(np.round(total8,5))  + f"(n_estimators = {rw8.n_estimators},  learning_rate = {rw8.learning_rate},  max_depth = {rw8.max_depth},  min_samples_leaf = {rw8.min_samples_leaf},  decayMode = {rw8.decayMode})\n"
    summaryString += str(np.round(total9,5))  + f"(n_estimators = {rw9.n_estimators},  learning_rate = {rw9.learning_rate},  max_depth = {rw9.max_depth},  min_samples_leaf = {rw9.min_samples_leaf},  decayMode = {rw9.decayMode})\n"
    summaryString += str(np.round(total10,5)) + f"(n_estimators = {rw10.n_estimators}, learning_rate = {rw10.learning_rate}, max_depth = {rw10.max_depth}, min_samples_leaf = {rw10.min_samples_leaf}, decayMode = {rw10.decayMode})\n"
    summaryString += str(np.round(total11,5)) + f"(n_estimators = {rw11.n_estimators}, learning_rate = {rw11.learning_rate}, max_depth = {rw11.max_depth}, min_samples_leaf = {rw11.min_samples_leaf}, decayMode = {rw11.decayMode})\n"
    summaryString += str(np.round(total12,5)) + f"(n_estimators = {rw12.n_estimators}, learning_rate = {rw12.learning_rate}, max_depth = {rw12.max_depth}, min_samples_leaf = {rw12.min_samples_leaf}, decayMode = {rw12.decayMode})\n"
    summaryString += str(np.round(total13,5)) + f"(n_estimators = {rw13.n_estimators}, learning_rate = {rw13.learning_rate}, max_depth = {rw13.max_depth}, min_samples_leaf = {rw13.min_samples_leaf}, decayMode = {rw13.decayMode})\n"
    summaryString += str(np.round(total14,5)) + f"(n_estimators = {rw14.n_estimators}, learning_rate = {rw14.learning_rate}, max_depth = {rw14.max_depth}, min_samples_leaf = {rw14.min_samples_leaf}, decayMode = {rw14.decayMode})\n"
    summaryString += str(np.round(total15,5)) + f"(n_estimators = {rw15.n_estimators}, learning_rate = {rw15.learning_rate}, max_depth = {rw15.max_depth}, min_samples_leaf = {rw15.min_samples_leaf}, decayMode = {rw15.decayMode})\n"
    summaryString += str(np.round(total16,5)) + f"(n_estimators = {rw16.n_estimators}, learning_rate = {rw16.learning_rate}, max_depth = {rw16.max_depth}, min_samples_leaf = {rw16.min_samples_leaf}, decayMode = {rw16.decayMode})\n"

    os.makedirs(os.path.dirname(f"savedModels/"), exist_ok=True)
    summaryFile = open(f"savedModels/{decayMode}ReweighterTuning.txt", "w")
    summaryFile.write(summaryString) 
    summaryFile.close()

if __name__ == "__main__":
    evaluateReweighter()
    # createStandardDistributions(feature="nTracks",  decayMode="pipi")
    # createStandardDistributions(feature="gamma_PT", decayMode="pipi")
    # # getAllWassTests(decayMode="kpi")
    # # getAllWassTests(decayMode="pipi")



