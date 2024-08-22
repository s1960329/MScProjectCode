
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

    plt.hist(  [SampleData[feature], MonteCarloData[feature]],
    weights =  [SampleData["NB0_Kpigamma_sw"], MCweights],
    label   =  ["LHCb", MClabel],
    color   =  [colors["red"],  MCcolor],
    **histStyle
    )

    plt.title("Feature Distributions")
    plt.xlabel(feature + " " + unitsDictionary[feature])
    plt.ylabel("Normalised Distribution")
    plt.grid(axis="both", linestyle="dashed", alpha=0.3)
    plt.legend()

    path = f"/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/imgs/standardFeatureDistributions/{decayMode}/"

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

def evaluateReweighter():

    inputFeatures = ["nTracks", "B_P", "gamma_PT", "daughter_neutral_PT", "B_Cone3_B_ptasy", "B_ETA"]

    SampleData     = pd.read_csv("data/SampleData.csv",      index_col=0)
    MonteCarloData = pd.read_csv("data/pipi/FullData.csv",   index_col=0)
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
    GBwithWeights.abbreviation = "Test"
    GBwithWeights.color        = colors["purple"]
    GBwithWeights.createInFull()

    print("- without weights")
    GBwithoutWeights = ForestClassifier(GradientBoostingClassifier(n_estimators=2500, learning_rate=0.005, max_depth=6, verbose=1), "EvaluateRewighterControl", inputFeatures, FullData)
    GBwithoutWeights.W_train = np.ones(len(GBwithoutWeights.W_train))
    GBwithoutWeights.W_test  = np.ones(len(GBwithoutWeights.W_test))
    GBwithoutWeights.abbreviation = "Control"
    GBwithoutWeights.color        = colors["black"]
    GBwithoutWeights.createInFull()

    plotROCcurves([GBwithWeights,GBwithoutWeights])

def findChiSquaredDifferenceInWeights(model, feature):

    model.createHist(feature)
    histogramDict = dict(zip(["SM","WMC","MC"], model.histY))
    
    WassBefore = wasserstein_distance(histogramDict["SM"], histogramDict["MC"])
    WassAfter  = wasserstein_distance(histogramDict["SM"], histogramDict["WMC"])

    print(f"\n {feature}")
    print("Before : ", WassBefore)
    print("After  : ", WassAfter)
    print("Diff   : ", WassAfter - WassBefore)

    return (WassAfter - WassBefore)



if __name__ == "__main__":

    pipiMonteCarloPath  =  "../data/dataROOT/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel.root"
    samplePath          =  "../data/dataROOT/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root"

    print("- pipi signal \n")
    pipi = GBReweight(monteCarloRootFile = pipiMonteCarloPath, sampleRootFile = samplePath, decayMode = "pipi")
    pipi.trainReweighter()
    pipi.computeWeights()

    total = 0
    for feature in allFeatures["any"]:
        total += findChiSquaredDifferenceInWeights(pipi, feature)

    print(total)


