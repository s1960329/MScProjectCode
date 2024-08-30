
import os
import joblib
import pandas as pd

from sklearn.metrics            import confusion_matrix, roc_curve, roc_auc_score, log_loss
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import KFold, train_test_split
from misc                       import *
from tensorflow.keras.models    import Sequential                         # type: ignore
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     # type: ignore
from sklearn.ensemble           import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier



class BaseClassifier():
    
    def __call__(self):
        return self.name

    def __init__(self, name, inputFeatures, inputFullData):
        self.abbreviation   = ""
        self.summaryString  = ""
        self.name           = name
        self.inputFeatures  = inputFeatures
        self.inputFullData  = inputFullData
        self.cut            = 0.5
        self.color          = "**"

        self.trainData, self.testData = train_test_split(inputFullData, test_size = 0.1, random_state=23)
    
        self.X_train        = self.trainData[self.inputFeatures]
        self.Y_train        = self.trainData["isSignal"]
        self.W_train        = self.trainData["weights"]

        self.X_test         = self.testData[self.inputFeatures]
        self.Y_test         = self.testData["isSignal"]
        self.W_test         = self.testData["weights"]

    def normaliseData(self, inputData = pd.DataFrame()):
        return inputData

    def predict(self, inputData = pd.DataFrame()):
        raise NotImplementedError("Must be overridden")
    
    def trainModel(self):
        raise NotImplementedError("Must be overridden")
    
    def classify(self, inputData = pd.DataFrame(), probCut = None):
        
        if probCut is None :
            probCut = self.cut

        if inputData.empty : 
            inputData = self.testData

        return roundUp(self.predict(inputData), probCut)
    
    def evaluateModel(self, inputData = pd.DataFrame(), probCut = None):

        if probCut is None : 
            probCut = self.cut
    
        self.confusionMatrixTest   = confusion_matrix(y_true=self.Y_test,   y_pred=roundUp(self.Y_predTest, probCut))
        self.confusionMatrixTrain  = confusion_matrix(y_true=self.Y_train,  y_pred=roundUp(self.Y_predTrain,probCut))
        
        self.TestScore       = roc_auc_score(self.Y_test,   self.Y_predTest, sample_weight=self.W_test)
        self.TestLogLoss     = log_loss(self.Y_test, self.Y_predTest, sample_weight=self.W_test)
        self.summaryString  += "\n"
        self.summaryString  += f"Test Data Log Loss  : {self.TestLogLoss}\n"
        self.summaryString  += f"Test Data Score     : {self.TestScore}\n"
        self.summaryString  += f"Test Confusion Matrix\n{self.confusionMatrixTest}\n"

        self.TrainScore      = roc_auc_score(self.Y_train, self.Y_predTrain, sample_weight=self.W_train)
        self.TrainLogLoss    = log_loss(self.Y_train, self.Y_predTrain, sample_weight=self.W_train)
        self.summaryString  += "\n"
        self.summaryString  += f"Train Data Log Loss : {self.TrainLogLoss}\n"
        self.summaryString  += f"Train Data Score    : {self.TrainScore}\n"
        self.summaryString  += f"Train Confusion Matrix\n{self.confusionMatrixTrain}\n"

        if not inputData.empty:
            self.confusionMatrixInput  = confusion_matrix(inputData["isSignal"],  roundUp(self.Y_predInput, probCut), sample_weight=inputData["weights"])
            self.InputScore            = roc_auc_score(inputData["isSignal"], self.Y_predInput, sample_weight=inputData["weights"])
            self.InputLogLoss          = log_loss(inputData["isSignal"], self.Y_predInput, sample_weight=inputData["weights"])
            self.summaryString        += "\n"
            self.summaryString        += f"Input Data Log Loss : {self.InputScore}\n"
            self.summaryString        += f"Input Data Score    : {self.InputLogLoss}\n"
            self.summaryString        += f"Input Confusion Matrix\n{self.confusionMatrixInput}\n"

        self.summaryString  += f"\n"
        self.summaryString  += f"Cut                 : {np.round(probCut,3)}\n"
        self.summaryString  += f"Input Variables     : {self.inputFeatures}\n"
    
    def saveModel(self, path="savedModels"):
        os.makedirs(os.path.dirname(f"{path}/{self.name[2:]}/"), exist_ok=True)
        summaryFile = open(f"{path}/{self.name[2:]}/{self.abbreviation}summary.txt", "w")
        summaryFile.write(self.summaryString) 
        summaryFile.close()
        joblib.dump(self, f"{path}/{self.name[2:]}/{self.name}.joblib")

    def createInFull(self, path="savedModels", inputData = pd.DataFrame()):
        print("\nTraining...\n")
        self.trainModel()
        print("\nEvaluating...\n")
        (self.cut, _ ) = getFoMandBestCut(np.linspace(0,1,1001), self, self.testData)
        self.evaluateModel(inputData)
        print("\nSaving...\n")
        self.saveModel(path)
        print("\nDone!\n")


class ForestClassifier(BaseClassifier):

    def __call__(self):
        return (self.name, self.model)
    
    def __init__(self, model, name, inputFeatures, inputFullData):
        BaseClassifier.__init__(self, name, inputFeatures, inputFullData)
        super().__init__(name, inputFeatures, inputFullData)

        self.model = model
        if   type(self.model) == GradientBoostingClassifier: self.abbreviation = "GB"
        elif type(self.model) == AdaBoostClassifier:         self.abbreviation = "AD"
        elif type(self.model) == RandomForestClassifier:     self.abbreviation = "RF"
        else :                                               self.abbreviation = "**"

        self.color = modelColors[self.abbreviation]
        self.name  = self.abbreviation + self.name

    def predict(self, inputData = pd.DataFrame()):

        if inputData.empty: 
            inputData = self.X_test

        return self.model.predict_proba(inputData[self.inputFeatures])[:, 1]

    def trainModel(self):
        fittedEstimator = self.model.fit(self.X_train, self.Y_train, sample_weight=self.W_train)
        return fittedEstimator

    def evaluateModel(self, inputData = pd.DataFrame(), probCut=None):

        self.Y_predTest  = self.predict(self.X_test)
        self.Y_predTrain = self.predict(self.X_train)

        if not inputData.empty: self.Y_predInput = self.predict(inputData)

        super().evaluateModel(inputData, probCut)

        self.summaryString += f"\n{self.model}\n"


class NeuralNetworkClassifier(BaseClassifier):

    def __call__(self):
        return (self.name, self.model, self.hiddenLayers)

    def __init__(self, hiddenLayers, name, inputFeatures, inputFullData):
        BaseClassifier.__init__(self, name, inputFeatures, inputFullData)
        super().__init__(name, inputFeatures, inputFullData)

        self.abbreviation   = "NN"
        self.color          = modelColors[self.abbreviation]
        self.name           = self.abbreviation + self.name
        self.epochs         = 100
        self.batchSize      = 32

        self.normaliser     = StandardScaler()
        self.normaliser.fit(self.inputFullData[inputFeatures], y=None, sample_weight=self.inputFullData["weights"])

        self.hiddenLayers   = hiddenLayers
        self.model = Sequential(self.hiddenLayers)
        self.model.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])

    def predict(self, inputData = pd.DataFrame()):
        if inputData.empty: inputData = self.X_test
        return self.model.predict(self.normaliseData(inputData[self.inputFeatures])).flatten()

    def trainModel(self):
        EarlyStoppingCallback = EarlyStopping(monitor="val_loss", patience=10)
        self.history = self.model.fit(self.normaliser.transform(self.X_train), self.Y_train, sample_weight=self.W_train, validation_data=(self.normaliser.transform(self.X_test), self.Y_test), callbacks=[EarlyStoppingCallback], batch_size=self.batchSize, epochs=self.epochs)

    def normaliseData(self, inputData):
        transData = pd.DataFrame(self.normaliser.transform(inputData[self.inputFeatures]), columns=self.inputFeatures)

        uniqueFeatures = list(set(inputData.columns) - set(self.inputFeatures))
        for uniqueFeature in uniqueFeatures:
            transData[uniqueFeature] = inputData[uniqueFeature]

        return transData

    def evaluateModel(self, inputData = pd.DataFrame(), probCut=None):
        
        self.Y_predTest  = self.predict(self.X_test)
        self.Y_predTrain = self.predict(self.X_train)

        if not inputData.empty: self.Y_predInput = self.predict(inputData)

        super().evaluateModel(inputData, probCut)

        self.summaryString += f"\n{NNSummaryToSring(self.model)}\n"
   

if __name__ == "__main__":
    pass






    





    


