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
    for feature in allFeatures["any"] : createReweightedHistogram(feature, decayMode="pipi")
    for feature in allFeatures["any"] : createReweightedHistogram(feature, decayMode="kpi")
    
    print(" --- Evaluating Reweighter")
    evaluateReweighter()

    print(" --- Formatting Data")
    combineSignalAndBackground(mode = "kpi")
    combineSignalAndBackground(mode = "pipi")

    createTestAndTrainData(mode = "kpi")
    createTestAndTrainData(mode = "pipi")

    createEvenTestAndTrainData(mode = "kpi")
    createEvenTestAndTrainData(mode = "pipi")

    createAllStandardDistributions()

    pipiTestData = pd.read_csv("/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/kpi/TestData.csv",  index_col=0)
    kpiTestData  = pd.read_csv("/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/pipi/TestData.csv", index_col=0)

    print(" --- Creating Classifiers")
    createBestClassifiers(name="Best", mode="pipi")
    createTestClassifiers(name="Test", mode="pipi")
    createBestClassifiers(name="Best", mode="kpi")
    createTestClassifiers(name="Test", mode="kpi")

    [RF,AD,GB,NN] = loadModels(name="Bestpipi")
    for feature in allFeatures["any"] : createBackgroundRemovalDistribution(GB, feature)

    print(" --- Load Models")
    ModelsBestpipi = loadModels(name="Bestpipi")
    ModelsTestpipi = loadModels(name="Testpipi")
    ModelsBestkpi  = loadModels(name="Bestkpi")
    ModelsTestkpi  = loadModels(name="Testkpi")

    print(" --- Creating Prediction Distributions")
    for model in ModelsBestpipi : plotSeperatedPredictions(model)
    for model in ModelsTestpipi : plotSeperatedPredictions(model)
    for model in ModelsBestkpi  : plotSeperatedPredictions(model)
    for model in ModelsTestkpi  : plotSeperatedPredictions(model)

    print(" --- Creating Efficiency Curves")
    for model in ModelsBestpipi : plotEfficiencies(model, inputData=pipiTestData)
    for model in ModelsTestpipi : plotEfficiencies(model, inputData=pipiTestData)
    for model in ModelsBestkpi  : plotEfficiencies(model, inputData=kpiTestData)
    for model in ModelsTestkpi  : plotEfficiencies(model, inputData=kpiTestData)

    print(" --- Creating Figures of Merit")
    for model in ModelsBestpipi : plotFigureOfMerit(model, inputData=pipiTestData)
    for model in ModelsTestpipi : plotFigureOfMerit(model, inputData=pipiTestData)
    for model in ModelsBestkpi  : plotFigureOfMerit(model, inputData=kpiTestData)
    for model in ModelsTestkpi  : plotFigureOfMerit(model, inputData=kpiTestData)

    print(" --- Creating ROC curves")
    plotROCcurves(ModelsBestpipi)
    plotROCcurves(ModelsTestpipi)
    plotROCcurves(ModelsBestkpi)
    plotROCcurves(ModelsTestkpi)


def TuneModels():

    inputFeatures   = ["nTracks", "B_P", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]
   
    names    = ["Tune1", "Tune2", "Tune3", "Tune4", "Tune5", "Tune6","Tune7","Tune8"]
    
    RFmodels = [RandomForestClassifier(n_estimators=50,    max_depth=3, verbose=1),
                RandomForestClassifier(n_estimators=100,   max_depth=3, verbose=1),
                RandomForestClassifier(n_estimators=500,   max_depth=5, verbose=1),
                RandomForestClassifier(n_estimators=1000,  max_depth=5, verbose=1),
                RandomForestClassifier(n_estimators=1500,  max_depth=6, verbose=1),
                RandomForestClassifier(n_estimators=2000,  max_depth=6, verbose=1),
                RandomForestClassifier(n_estimators=1500,  verbose=1),
                RandomForestClassifier(n_estimators=2000,  verbose=1)]
    
    ADmodels = [AdaBoostClassifier(n_estimators=50,   learning_rate=1.0),
                AdaBoostClassifier(n_estimators=100,  learning_rate=1.0),
                AdaBoostClassifier(n_estimators=500,  learning_rate=0.1),
                AdaBoostClassifier(n_estimators=1000, learning_rate=0.1),
                AdaBoostClassifier(n_estimators=1500, learning_rate=0.01),
                AdaBoostClassifier(n_estimators=2000, learning_rate=0.01),
                AdaBoostClassifier(n_estimators=1500, learning_rate=1.0),
                AdaBoostClassifier(n_estimators=2000, learning_rate=1.0)]
    
    GBmodels = [GradientBoostingClassifier(n_estimators=50,  learning_rate=0.1,    max_depth=3,verbose=1),
                GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,    max_depth=3,verbose=1),
                GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,   max_depth=5,verbose=1),
                GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01,   max_depth=5,verbose=1),
                GradientBoostingClassifier(n_estimators=1500,learning_rate=0.001,  max_depth=6,verbose=1),
                GradientBoostingClassifier(n_estimators=2000,learning_rate=0.001,  max_depth=6,verbose=1),
                GradientBoostingClassifier(n_estimators=1500,learning_rate=0.001,  max_depth=None,verbose=1),
                GradientBoostingClassifier(n_estimators=2000,learning_rate=0.001,  max_depth=None,verbose=1)
                ]
    
    NNmodels = [ 
                 [Dense(1, input_shape = (len(inputFeatures),), activation="sigmoid", kernel_initializer="random_normal")],
                
                 [Dense(len(inputFeatures),   input_shape = (len(inputFeatures),),   activation="relu",      kernel_initializer="random_normal"),
                  Dense(1,                    input_shape = (len(inputFeatures),),   activation="sigmoid",   kernel_initializer="random_normal") ],
                 
                 [Dense(6*len(inputFeatures), input_shape = (len(inputFeatures),),   activation="relu",      kernel_initializer="random_normal"),
                  Dense(len(inputFeatures),   input_shape = (6*len(inputFeatures),), activation="relu",      kernel_initializer="random_normal"),
                  Dense(1,                    input_shape = (len(inputFeatures),),   activation="sigmoid",   kernel_initializer="random_normal") ],

                 [Dense(6*len(inputFeatures), input_shape = (len(inputFeatures),),   activation="relu",      kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(len(inputFeatures),   input_shape = (6*len(inputFeatures),), activation="relu",      kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(1,                    input_shape = (len(inputFeatures),),   activation="sigmoid",   kernel_initializer="random_normal") ],
                  
                 [Dense(len(inputFeatures),    input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                  Dense(len(inputFeatures)*12, input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                  Dense(len(inputFeatures),    input_shape = (len(inputFeatures)*12,), activation="relu", kernel_initializer="random_normal"),
                  Dense(1,                     input_shape = (len(inputFeatures),),    activation="sigmoid",    kernel_initializer="random_normal")],

                 [Dense(len(inputFeatures),    input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(len(inputFeatures)*12, input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(len(inputFeatures),    input_shape = (len(inputFeatures)*12,), activation="relu", kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(1,                      input_shape = (len(inputFeatures),),    activation="sigmoid",    kernel_initializer="random_normal")],
                  
                 [Dense(len(inputFeatures)*12,  input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                  Dense(len(inputFeatures)*120, input_shape = (len(inputFeatures)*12,),    activation="relu", kernel_initializer="random_normal"),
                  Dense(len(inputFeatures)*12,  input_shape = (len(inputFeatures)*120,), activation="relu", kernel_initializer="random_normal"),
                  Dense(1,                      input_shape = (len(inputFeatures)*12,),    activation="sigmoid",    kernel_initializer="random_normal")],
                
                 [Dense(len(inputFeatures)*12,  input_shape = (len(inputFeatures),),    activation="relu", kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(len(inputFeatures)*120, input_shape = (len(inputFeatures)*12,),    activation="relu", kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(len(inputFeatures)*12,  input_shape = (len(inputFeatures)*120,), activation="relu", kernel_initializer="random_normal"),
                  Dropout(0.2),
                  Dense(1,                      input_shape = (len(inputFeatures)*12,),    activation="sigmoid",    kernel_initializer="random_normal")]
                
                ]

    modelDicts = [ {"RF" : RF, "AD" : AD, "GB" :GB , "NN" : NN} for RF, AD, GB, NN in zip(RFmodels,ADmodels,GBmodels,NNmodels) ]

    # for name, d in zip(names, modelDicts):
    #     createVariableClassifiers(name, "pipi", d)

    for name, d in zip(names, modelDicts):
        createVariableClassifiers(name, "kpi", d)


if __name__ == "__main__":

    models = loadModels(name="Bestkpi")

    for model in models :
        for feature in allFeatures["any"] : 
            createBackgroundRemovalDistribution(model, feature)