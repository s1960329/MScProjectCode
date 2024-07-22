
import os
import joblib
import pandas as pd

from sklearn.metrics            import confusion_matrix, roc_curve, roc_auc_score
from misc                       import *
from tensorflow.keras.models    import Sequential     # type: ignore
from tensorflow.keras.layers    import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.ensemble           import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier


class NeuralNetworkClassifier():

    def __call__(self):
        return (self.name, self.NNmodel)

    def __init__(self, name, hiddenLayers, inputVariables, trainData, testData):
        self.name           = name
        self.hiddenLayers   = hiddenLayers

        self.X_train        = trainData[inputVariables]
        self.Y_train        = trainData["isSignal"]
        self.W_train        = trainData["weights"]

        self.X_test         = testData[inputVariables]
        self.Y_test         = testData["isSignal"]
        self.W_test         = testData["weights"]
        
        self.createModel()

    def createModel(self):
        self.NNmodel = Sequential(self.hiddenLayers)
        self.NNmodel.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])

    def trainModel(self):
        EarlyStoppingCallback = EarlyStopping(monitor="val_loss", patience=10)
        history = self.NNmodel.fit(self.X_train, self.Y_train, sample_weight=self.W_train, validation_data=(self.X_test, self.Y_test), batch_size=10, epochs=500, callbacks=[EarlyStoppingCallback])
        return history
    
    def evaluateModel(self):
        Y_predTest      = roundUp(self.NNmodel.predict(self.X_test).flatten())
        Y_predTrain     = roundUp(self.NNmodel.predict(self.X_train).flatten())
        confusionMatrix = confusion_matrix(y_true=self.Y_test, y_pred=Y_predTest)
        print(f"Confusion Matrix\n{confusionMatrix}\n")
        print(f"Test Data Score     : {roc_auc_score(self.Y_test,  Y_predTest)}\n")
        print(f"Training Data Score : {roc_auc_score(self.Y_train, Y_predTrain)}\n")
    
    def createROCcurve(self):
        Y_predTest = self.NNmodel.predict(self.X_test).flatten()
        falsePositiveRate, truePositiveRate, _ = roc_curve(self.Y_test, Y_predTest)
        return (falsePositiveRate, truePositiveRate)

    def predict(self, inputData):
        return self.NNmodel.predict(inputData).flatten()

    def saveModel(self):
        os.mkdir(f"savedModels/{name}") 
        joblib.dump(self, f"savedModels/{name}/NN{name}.joblib")



class ForestClassifiers():

    def __call__(self):
        return self.name

    def __init__(self, name, classifierType, inputVariables):
        self.name = name
        self.classifierType = classifierType
        self.inputVariables = inputVariables
        
        self.FullData       = pd.read_csv(f"{dataPath}/dataLearn/FullDataClassified.csv", index_col=0)
        self.signalData     = pd.read_csv(f"{dataPath}/dataLearn/SignalData.csv",index_col=0)
        self.backgroundData = pd.read_csv(f"{dataPath}/dataLearn/BackgroundData.csv",index_col=0)
        train               = pd.read_csv(f"{dataPath}/dataLearn/TrainingData.csv",index_col=0)
        test                = pd.read_csv(f"{dataPath}/dataLearn/TestingData.csv",index_col=0)

        self.X_train        = train[inputVariables]
        self.Y_train        = train["isSignal"]
        self.W_train        = train["weights"]

        self.X_test         = test[inputVariables]
        self.Y_test         = test["isSignal"]
        self.W_test         = test["weights"]


    def trainModel(self):
        print(f"\n- Training {self.classifierType} ... \n")
        fittedEstimator = self.classifierType.fit(self.X_train, self.Y_train, sample_weight=self.W_train)
        return fittedEstimator
    
    def evaluateModel(self):
        Y_predTest  = self.classifierType.predict(self.X_test)
        Y_predTrain = self.classifierType.predict(self.X_train)

        confusionMatrix = confusion_matrix(y_true=self.Y_test, y_pred=Y_predTest)
        print(f"Confusion Matrix\n{confusionMatrix}\n")
        print(f"Test Data Score     : {roc_auc_score(self.Y_test,  Y_predTest)}\n")
        print(f"Training Data Score : {roc_auc_score(self.Y_train, Y_predTrain)}\n")
        AFfalsePositiveRate, AFtruePositiveRate, threshold = createTreeROCcurve(self.classifierType, self.X_test, self.Y_test)
        
        return (AFfalsePositiveRate, AFtruePositiveRate, threshold)

    def summary(self):
        return self.classifierType
    
    def predict(self, inputData):
        return self.classifierType.predict_proba(inputData)[:, 1]
    
    def savePredictionProbabilities(self):
        self.FullData[f"SignalProb{name}"] = self.predict(self.FullData[self.inputVariables])
        self.FullData.to_csv(f"{dataPath}/dataLearn/FullDataClassified.csv")
        return self.FullData
    
    def saveModel(self, filename):
        os.makedirs(os.path.dirname(f"{modelPath}/{name}"), exist_ok=True)
        joblib.dump(self, f"{modelPath}/{name}/{filename}.joblib")

    def executeAll(self):
        self.trainModel()
        self.evaluateModel()
        self.savePredictionProbabilities()

class CombinedClassifier():

    def __call__(self):
        pass

    def __init__(self, name, inputVariables):


        self.NN = NeuralNetworkClassifier(name, hiddenLayers, inputVariables)
        self.GB = ForestClassifiers(      name, GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3 ),         inputVariables)
        self.RF = ForestClassifiers(      name, RandomForestClassifier(    n_estimators=50,                     max_depth=3 ),         inputVariables)
        self.AD = ForestClassifiers(      name, AdaBoostClassifier(        n_estimators=500, learning_rate=0.1, algorithm="SAMME.R" ), inputVariables)
    
        self.allModels = [self.NN, self.GB, self.RF, self.AD]
    
    def trainModel(self):
        for model in self.allModels:
            model.trainModel()

    def predict(self, inputData):
        predicitions = [model.predict(inputData) for model in self.allModels]
        return np.mean(predicitions)

    def evaluateModel(self):
        Y_predTest  = self.predict(self.GB.X_test)
        Y_predTrain = self.predict(self.GB.X_train)

        confusionMatrix = confusion_matrix(y_true=self.GB.Y_test, y_pred=Y_predTest)
        print(f"Confusion Matrix\n{confusionMatrix}\n")
        print(f"Test Data Score     : {roc_auc_score(self.GB.Y_test,  Y_predTest)}\n")
        print(f"Training Data Score : {roc_auc_score(self.GB.Y_train, Y_predTrain)}\n")


if __name__ == "__main__":

    trainDataNorm       = pd.read_csv("data/Norm/TestNorm.csv"  , index_col=0)
    testDataNorm        = pd.read_csv("data/Norm/TrainNorm.csv" , index_col=0)


    name           =   "Default"
    inputVariables =   ['nTracks', 'B_Cone3_B_ptasy', 'B_ETA', 'B_MINIPCHI2', 'piminus_PT', 'piminus_IP_OWNPV', 'daughterplus_PT', 'daughterplus_IP_OWNPV']
    hiddenLayers   =   [Dense(len(inputVariables),    input_shape = (len(inputVariables),),    activation="relu"),
                        Dropout(0.025),
                        Dense(len(inputVariables)*10, input_shape = (len(inputVariables),),    activation="relu"),
                        Dropout(0.025),
                        Dense(len(inputVariables),    input_shape = (len(inputVariables)*10,), activation="relu"),
                        Dropout(0.025),
                        Dense(1,                      input_shape = (len(inputVariables),),    activation="sigmoid")]
    

    CC = NeuralNetworkClassifier(name, hiddenLayers, inputVariables, trainDataNorm, testDataNorm)
    CC.trainModel()
    CC.evaluateModel()
    CC.saveModel()

