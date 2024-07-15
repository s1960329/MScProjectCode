import joblib
import os
import pandas                   as     pd
import numpy                    as     np
import matplotlib.pyplot        as     plt


from sklearn.metrics            import confusion_matrix, roc_curve, roc_auc_score
from misc                       import fileNamesCSV, uniqueVariables, imagePath
from misc                       import formatForClassification, normaliseData, createTreeROCcurve, NNSummaryToSring, roundUp
from sklearn.ensemble           import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from tensorflow.keras.models    import Sequential, load_model  # type: ignore
from tensorflow.keras.layers    import Input, Dense, Dropout   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping           # type: ignore
    
# ["nTracks", "B_P", "gamma_PT", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]
# ["gamma_PT","nTracks","piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]
# ["nTracks", "B_P", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV"]
# ["nTracks", "B_P", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]

outputString      = ""
ExperimentName    = "Combine"
signalPath        = fileNamesCSV["pipi"]
backgroundPath    = fileNamesCSV["kpisb"]

isSignalEvenSplit = True
trainingVariables = ['nTracks', 'B_P', 'B_Cone3_B_ptasy', 'B_ETA', 'B_MINIPCHI2', 'B_SmallestDeltaChi2OneTrack', 'B_FD_OWNPV', 'piminus_PT', 'piminus_IP_OWNPV', 'daughter_neutral_PT', 'daughter_neutral_IP_OWNPV', 'daughterplus_PT', 'daughterplus_IP_OWNPV']
NNcutOffProb      = 0.5

RandomForestObj   = RandomForestClassifier(     n_estimators=50,  max_depth=3)
GradientBoostObj  = GradientBoostingClassifier( n_estimators=800, learning_rate=0.012, max_depth=3)
AdaBoostObj       = AdaBoostClassifier(         n_estimators=500, learning_rate=0.1, algorithm="SAMME.R") 


hiddenLayers      = [Dense(len(trainingVariables),    input_shape = (len(trainingVariables),),    activation="relu"),
                     Dense(len(trainingVariables)*10, input_shape = (len(trainingVariables),),    activation="relu"),
                     Dense(len(trainingVariables),    input_shape = (len(trainingVariables)*10,), activation="relu"),
                     Dense(1,                         input_shape = (len(trainingVariables),),    activation="sigmoid")]


((X_train,Y_train,W_train),(X_test,Y_test,W_test)) = formatForClassification(trainingVariables, signalPath, backgroundPath, isSignalEvenSplit)
X_trainNormalised = normaliseData(X_train)
X_testNormalised  = normaliseData(X_test)


class NeuralNetworkClassifier():

    def __call__(self):
        return self.NNmodel

    def __init__(self,  hiddenLayers):
        self.hiddenLayers = hiddenLayers
        self.CreateModel()

    def CreateModel(self):

        global outputString

        self.NNmodel = Sequential()
        for layer in self.hiddenLayers:
            self.NNmodel.add(layer)

        self.NNmodel.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])
        outputString += NNSummaryToSring(self.NNmodel)

    def ExecuteAll(self):
        history = self.TrainModel()
        (NNfalsePositiveRate, NNtruePositiveRate, threshold) = self.EvaluateModel()
        
        return (history, NNfalsePositiveRate, NNtruePositiveRate, threshold)

    def TrainModel(self):
        bs = int(len(X_testNormalised) / 1000)
        EarlyStoppingCallback = EarlyStopping(monitor="val_loss", patience=3)
        history = self.NNmodel.fit(X_trainNormalised, Y_train, sample_weight=W_train, validation_data=(X_testNormalised, Y_test), batch_size=bs, epochs=500, callbacks=[EarlyStoppingCallback])
        
        return history
    
    def EvaluateModel(self):

        global outputString

        Y_predTest  = self.NNmodel.predict(X_testNormalised)
        Y_predTrain = self.NNmodel.predict(X_trainNormalised)

        NNfalsePositiveRate, NNtruePositiveRate, threshold = roc_curve(Y_test, Y_predTest)
        confusionMatrixNN = confusion_matrix(y_true=Y_test, y_pred=np.round(Y_predTest))
        
        outputString += f"\nConfusion Matrix \n{confusionMatrixNN}\n"
        outputString += f"Test Data Score     : {roc_auc_score(Y_test,  Y_predTest)}\n"
        outputString += f"Training Data Score : {roc_auc_score(Y_train, Y_predTrain)}\n"
        return (NNfalsePositiveRate, NNtruePositiveRate, threshold)


class AssortedForestClassifier():

    def __call__(self):
        return self.classifierType

    def __init__(self, classifierType):
        self.classifierType = classifierType

    def ExecuteAll(self):
        fittedEstimator = self.TrainModel()
        (AFfalsePositiveRate, AFtruePositiveRate, threshold) = self.EvaluateModel()
        
        return (fittedEstimator, AFfalsePositiveRate, AFtruePositiveRate, threshold)

    def TrainModel(self):

        global outputString

        print(f"\n- Training {self.classifierType} ... \n")
        fittedEstimator = self.classifierType.fit(X_train, Y_train, sample_weight=W_train)
        outputString += f"{self.classifierType}\n"
        return fittedEstimator

    def EvaluateModel(self):

        global outputString

        Y_predTest  = self.classifierType.predict(X_testNormalised)
        Y_predTrain = self.classifierType.predict(X_trainNormalised)

        confusionMatrix = confusion_matrix(y_true=Y_test, y_pred=Y_predTest)
        outputString += f"Confusion Matrix\n{confusionMatrix}\n"
        outputString += f"Test Data Score     : {roc_auc_score(Y_test,  Y_predTest)}\n"
        outputString += f"Training Data Score : {roc_auc_score(Y_train, Y_predTrain)}\n"
        AFfalsePositiveRate, AFtruePositiveRate, threshold = createTreeROCcurve(self.classifierType, X_test, Y_test)
        
        return (AFfalsePositiveRate, AFtruePositiveRate, threshold)


class CombinedClassifier():

    def __init__(self):
        self.RF = AssortedForestClassifier(classifierType = RandomForestObj  )
        self.GB = AssortedForestClassifier(classifierType = GradientBoostObj )
        self.AD = AssortedForestClassifier(classifierType = AdaBoostObj      )
        self.NN = NeuralNetworkClassifier(hiddenLayers)

        self.RF.TrainModel()
        self.GB.TrainModel()
        self.AD.TrainModel()
        self.NN.TrainModel()

    def predict(self):

        NNprediction = self.NN.NNmodel.predict(X_testNormalised).flatten()
        ADprediction = self.GB.classifierType.predict_proba(X_test)[:, 1]
        GBprediction = self.AD.classifierType.predict_proba(X_test)[:, 1]  
        RFprediction = self.RF.classifierType.predict_proba(X_test)[:, 1]

        votesPerEntry = zip(NNprediction, ADprediction, GBprediction, RFprediction)
        
        combinedPrediction = np.array([sum(entry)/len(entry) for entry in votesPerEntry])

        return combinedPrediction



def trainAllClassifiers(name=ExperimentName):

    global outputString
    savedModelsPath = f"savedModels/{name}/"
    os.makedirs(os.path.dirname(savedModelsPath), exist_ok=True)

    RF = AssortedForestClassifier(classifierType = RandomForestObj)
    (RFfittedEstimator, RFfalsePositiveRate, RFtruePositiveRate, RFthreshold) = RF.ExecuteAll()
    joblib.dump(RF, f"savedModels/{name}/RandomForest_{name}.joblib")
    outputString += "\n"

    GB = AssortedForestClassifier(classifierType = GradientBoostObj )
    (GBfittedEstimator, GBfalsePositiveRate, GBtruePositiveRate, GBthreshold) = GB.ExecuteAll()
    joblib.dump(GB, f"savedModels/{name}/GradientBoosting_{name}.joblib")
    outputString += "\n"

    AD = AssortedForestClassifier(classifierType = AdaBoostObj )
    (ADfittedEstimator, ADfalsePositiveRate, ADtruePositiveRate, ADthreshold) = AD.ExecuteAll()
    joblib.dump(AD, f"savedModels/{name}/AdaBoost_{name}.joblib")
    outputString += "\n"

    NN = NeuralNetworkClassifier(hiddenLayers)
    (RFfittedEstimator, NNfalsePositiveRate, NNtruePositiveRate, RFthreshold) = NN.ExecuteAll()
    NN.NNmodel.save(f"savedModels/{name}/NNmodel_{name}.keras")
    outputString += "\n"

    #    

    outputString += "Combined Classifier\n"    
    NNpredictionTest = NN.NNmodel.predict(X_testNormalised).flatten()
    ADpredictionTest = GB.classifierType.predict_proba(X_test)[:, 1]
    GBpredictionTest = AD.classifierType.predict_proba(X_test)[:, 1]  
    RFpredictionTest = RF.classifierType.predict_proba(X_test)[:, 1]

    votesPerEntryTest = zip(NNpredictionTest*0.5, ADpredictionTest*0.0, GBpredictionTest*0.5, RFpredictionTest*0.0)
    combinedPredictionTest = np.array([sum(entry) for entry in votesPerEntryTest])

    NNpredictionTrain = NN.NNmodel.predict(X_trainNormalised).flatten()
    ADpredictionTrain = GB.classifierType.predict_proba(X_train)[:, 1]
    GBpredictionTrain = AD.classifierType.predict_proba(X_train)[:, 1]  
    RFpredictionTrain = RF.classifierType.predict_proba(X_train)[:, 1]

    votesPerEntryTrain = zip(NNpredictionTrain*0.5, ADpredictionTrain*0.0, GBpredictionTrain*0.5, RFpredictionTrain*0.0)
    combinedPredictionTrain = np.array([sum(entry) for entry in votesPerEntryTrain])


    confusionMatrix = confusion_matrix(y_true=Y_test, y_pred=roundUp(combinedPredictionTest))
    outputString += f"Confusion Matrix\n{confusionMatrix}\n"
    outputString += f"Test Data Score     : {roc_auc_score(Y_test, combinedPredictionTest)}\n"
    outputString += f"Train Data Score    : {roc_auc_score(Y_train, combinedPredictionTrain)}\n"

    COfalsePositiveRate, COtruePositiveRate, COthreshold = roc_curve(Y_test, combinedPredictionTest)

    #
    
    plt.figure(figsize=(10, 8))
    plt.plot(COfalsePositiveRate, COtruePositiveRate, label="Combined Prediction")
    plt.plot(NNfalsePositiveRate, NNtruePositiveRate, label="Neural Network")
    plt.plot(RFfalsePositiveRate, RFtruePositiveRate, label="Random Forest")
    plt.plot(GBfalsePositiveRate, GBtruePositiveRate, label="Gradient Boost")
    plt.plot(ADfalsePositiveRate, ADtruePositiveRate, label="Adaboost")
    plt.plot([0, 1], [0, 1], "k--")  # Added diagonal line for random classifier
    plt.xlabel("False Positive Rate") 
    plt.ylabel("True Positive Rate")  
    plt.title(f"{name} ROC Curve")

    modelInfoText = f"\nisEvenSplit = {isSignalEvenSplit}\ntrainingVariables = {trainingVariables}\n"
    plt.text(x=0, y=8, s=modelInfoText, fontsize=8)

    plt.legend(fontsize=8)
    plt.savefig(f"savedModels/{name}/ROC_{name}.png", dpi=227)
    plt.close()

    outputString += modelInfoText

    with open(f"{savedModelsPath}summary_{name}.txt", "w") as text_file:
        text_file.write(outputString)

    print("\nFinished!\n")


if __name__ == "__main__":
    trainAllClassifiers()

    










