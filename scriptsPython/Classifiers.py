
import os
import joblib
import pandas as pd

from sklearn.metrics            import confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing      import MinMaxScaler
from misc                       import *
from tensorflow.keras.models    import Sequential                         # type: ignore
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping                      # type: ignore
from sklearn.ensemble           import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier


#TODO add Kfolds to the data


class BaseClassifier():

    def __init__(self, name, inputVariables, trainData, testData):
        self.abbreviation   = ""
        self.summaryString  = ""
        self.name           = name
        self.inputVariables = inputVariables
        self.cut            = 0.5

        self.X_train        = trainData[inputVariables]
        self.Y_train        = trainData["isSignal"]
        self.W_train        = trainData["weights"]

        self.X_test         = testData[inputVariables]
        self.Y_test         = testData["isSignal"]
        self.W_test         = testData["weights"]

    def predict(self, inputData = None):
        raise NotImplementedError("Must be overridden")
    
    def trainModel(self):
        raise NotImplementedError("Must be overridden")
    
    def classify(self, inputData = None, probCut = None):
        if probCut is None : probCut = self.cut
        return roundUp(self.predict(inputData), probCut)
    
    def evaluateModel(self, inputData = None, probCut = None):

        if probCut == None: probCut = self.cut

        Y_predTest                 = self.predict(self.X_test)
        Y_predTrain                = self.predict(self.X_train)

        self.confusionMatrixTest   = confusion_matrix(y_true=self.Y_test,   y_pred=roundUp(Y_predTest, probCut))
        self.confusionMatrixTrain  = confusion_matrix(y_true=self.Y_train,  y_pred=roundUp(Y_predTrain,probCut))
        
        self.TestScore       = roc_auc_score(self.Y_test,   Y_predTest)
        self.summaryString  += "\n"
        self.summaryString  += f"Test Data Score     : {self.TestScore}\n"
        self.summaryString  += f"Test Confusion Matrix\n{self.confusionMatrixTest}\n"

        self.TrainScore      = roc_auc_score(self.Y_train, Y_predTrain)
        self.summaryString  += "\n"
        self.summaryString  += f"Train Data Score    : {self.TrainScore}\n"
        self.summaryString  += f"Train Confusion Matrix\n{self.confusionMatrixTrain}\n"

        if inputData is not None:
            Y_predInput                = self.predict(inputData)
            self.confusionMatrixInput  = confusion_matrix(y_true=inputData["isSignal"],  y_pred=roundUp(Y_predInput, probCut))
            self.InputScore            = roc_auc_score(inputData["isSignal"], Y_predInput)
            self.summaryString        += "\n"
            self.summaryString        += f"Input Data Score    : {roc_auc_score(inputData["isSignal"], Y_predInput)}\n"
            self.summaryString        += f"Input Confusion Matrix\n{self.confusionMatrixInput}\n"

        self.summaryString  += f"\n"
        self.summaryString  += f"Input Variables     : {self.inputVariables}\n"
    
    def createROCcurve(self, inputData = None):

        if inputData is None: 
            inputData = self.X_test
            Y_truth   = self.Y_test
        else:
            Y_truth = inputData["isSignal"]

        Y_predTest = self.predict(inputData)
        falsePositiveRate, truePositiveRate, _ = roc_curve(y_true=Y_truth, y_score=Y_predTest)
        return (falsePositiveRate, truePositiveRate)

    def saveModel(self):
        os.makedirs(os.path.dirname(f"savedModels/{self.name[2:]}/"), exist_ok=True)
        summaryFile = open(f"savedModels/{self.name[2:]}/{self.abbreviation}summary.txt", "w")
        summaryFile.write(self.summaryString) 
        summaryFile.close()
        joblib.dump(self, f"savedModels/{self.name[2:]}/{self.name}.joblib")

    def createInFull(self, inputData = None):
        print("\nTraining...\n")
        self.trainModel()
        print("Evaluating...\n")
        self.evaluateModel(inputData)
        print("Saving...\n")
        self.saveModel()
        print("Done!\n")
    

class NeuralNetworkClassifier(BaseClassifier):

    def __call__(self):
        return (self.name, self.model, self.hiddenLayers)

    def __init__(self, hiddenLayers, name, inputVariables, trainData, testData):
        BaseClassifier.__init__(self, name, inputVariables, trainData, testData)
        super().__init__(name, inputVariables, trainData, testData)

        self.abbreviation   = "NN"
        self.name           = self.abbreviation + self.name

        self.normaliser     = MinMaxScaler()
        ScaledTrainData     = self.normaliser.fit_transform(trainData[inputVariables])
        ScaledTestData      = self.normaliser.fit_transform( testData[inputVariables])
        
        self.X_train        = pd.DataFrame(ScaledTrainData, columns=inputVariables)
        self.X_test         = pd.DataFrame(ScaledTestData,  columns=inputVariables)

        self.hiddenLayers   = hiddenLayers
        self.model = Sequential(self.hiddenLayers)
        self.model.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])

    def predict(self, inputData = None):
        if inputData is None: inputData = self.X_test
        transformedData = self.normaliser.fit_transform(inputData[self.inputVariables])
        return self.model.predict(transformedData).flatten()

    def trainModel(self):
        EarlyStoppingCallback = EarlyStopping(monitor="val_loss", patience=10)
        self.history = self.model.fit(self.X_train, self.Y_train, sample_weight=self.W_train, validation_data=(self.X_test, self.Y_test), batch_size=32, epochs=100, callbacks=[EarlyStoppingCallback])

    def evaluateModel(self, inputData = None):
        super().evaluateModel(inputData)
        self.summaryString += f"\n{NNSummaryToSring(self.model)}\n"
   

class ForestClassifier(BaseClassifier):

    def __call__(self):
        return (self.name, self.model)
    
    def __init__(self, model, name, inputVariables, trainData, testData):
        BaseClassifier.__init__(self, name, inputVariables, trainData, testData)
        super().__init__(name, inputVariables, trainData, testData)

        self.model = model
        if   type(self.model) == GradientBoostingClassifier: self.abbreviation = "GB"
        elif type(self.model) == AdaBoostClassifier:         self.abbreviation = "AD"
        elif type(self.model) == RandomForestClassifier:     self.abbreviation = "RF"
        else :                                               self.abbreviation = "**"

        self.name  = self.abbreviation + self.name

    def predict(self, inputData = None):
        if inputData is None: inputData = self.X_test
        return self.model.predict_proba(inputData[self.inputVariables])[:, 1]

    def trainModel(self):
        fittedEstimator = self.model.fit(self.X_train, self.Y_train, sample_weight=self.W_train)
        return fittedEstimator
    
    def evaluateModel(self, inputData = None):
        super().evaluateModel(inputData)
        self.summaryString += f"\n{self.model}\n"


def createBestClassifiers(mode = "pipi"):

    name            = f"Best{mode}"
    inputVariables  = ["nTracks", "B_P", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]
    testData        = pd.read_csv(f"data/{mode}/TestData.csv",  index_col=0)
    trainData       = pd.read_csv(f"data/{mode}/TrainData.csv", index_col=0)

    GlobalParams    = { "name"           : name,
                        "inputVariables" : inputVariables,
                        "testData"       : testData,
                        "trainData"      : trainData}

    hiddenLayers    =  [BatchNormalization(axis = 1),
                        Dense(len(inputVariables),    input_shape = (len(inputVariables),),    activation="relu",    kernel_initializer="random_normal"),
                        Dropout(0.05),
                        BatchNormalization(axis = 1),
                        Dense(len(inputVariables)*12, input_shape = (len(inputVariables),),    activation="relu",    kernel_initializer="random_normal"),
                        Dropout(0.05),
                        BatchNormalization(axis = 1),
                        Dense(len(inputVariables),    input_shape = (len(inputVariables)*12,), activation="relu",    kernel_initializer="random_normal"),
                        Dropout(0.05),
                        BatchNormalization(axis = 1),
                        Dense(1,                      input_shape = (len(inputVariables),),    activation="sigmoid", kernel_initializer="random_normal")]

    print("- Random Forest")
    RF = ForestClassifier(RandomForestClassifier(n_estimators=1500, max_depth=6, verbose=1), **GlobalParams)
    RF.createInFull()
    
    print("- AdaBoost")
    AD = ForestClassifier(AdaBoostClassifier(n_estimators=500, learning_rate=0.1), **GlobalParams)
    AD.createInFull()

    print("- Neural Network")
    NN = NeuralNetworkClassifier(hiddenLayers, **GlobalParams)
    NN.createInFull()

    print("- Gradient Boost")
    GB = ForestClassifier(GradientBoostingClassifier(n_estimators=800, learning_rate=0.012, max_depth=4, verbose=1), **GlobalParams)
    GB.createInFull()


def createTestModel(mode = "pipi"):
    name            = f"test{mode}"
    inputVariables  = ["nTracks", "B_P", "B_Cone3_B_ptasy", "B_ETA", "B_MINIPCHI2", "B_SmallestDeltaChi2OneTrack", "B_FD_OWNPV", "piminus_PT", "piminus_IP_OWNPV","daughter_neutral_PT", "daughter_neutral_IP_OWNPV", "daughterplus_PT","daughterplus_IP_OWNPV"]
    testData        = pd.read_csv(f"data/{mode}/TestData.csv",  index_col=0)
    trainData       = pd.read_csv(f"data/{mode}/TrainData.csv", index_col=0)

    hiddenLayers    =  [BatchNormalization(axis = 1),
                        Dense(len(inputVariables),    input_shape = (len(inputVariables),),    activation="sigmoid",    kernel_initializer="random_normal"),
                        Dropout(0.05),
                        BatchNormalization(axis = 1),
                        Dense(len(inputVariables)*12, input_shape = (len(inputVariables),),    activation="sigmoid",    kernel_initializer="random_normal"),
                        Dropout(0.05),
                        BatchNormalization(axis = 1),
                        Dense(len(inputVariables),    input_shape = (len(inputVariables)*12,), activation="sigmoid",    kernel_initializer="random_normal"),
                        Dropout(0.05),
                        BatchNormalization(axis = 1),
                        Dense(1,                      input_shape = (len(inputVariables),),    activation="sigmoid", kernel_initializer="random_normal")]

    model = NeuralNetworkClassifier(hiddenLayers, name, inputVariables, trainData, testData)
    model.createInFull()


if __name__ == "__main__":
    createTestModel(mode="pipi")



    


