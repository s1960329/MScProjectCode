import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.metrics            import confusion_matrix, roc_curve, roc_auc_score
from misc                       import fileNamesCSV, uniqueVariables, imagePath, formatForClassification, normaliseData
from sklearn.ensemble           import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from tensorflow.keras.models    import Sequential     # type: ignore
from tensorflow.keras.layers    import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    

((X_train,Y_train,W_train),(X_test,Y_test,W_test)) = formatForClassification()
X_trainNormalised = normaliseData(X_train)
X_testNormalised  = normaliseData(X_test)




class NeuralNetworkClassifier():

    def __call__(self):
        pass

    def __init__(self):
        pass

    def CreateNNModel(self):
        model = Sequential()
        model.add(Dense(120,  input_shape = (6,),   activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(120,  input_shape = (120,), activation="relu"))
        model.add(Dense(1,    input_shape = (120,), activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])
        model.summary()

        EarlyStoppingCallback = EarlyStopping(monitor="val_loss", patience=3)
        history = model.fit(X_trainNormalised, Y_train, sample_weight=W_train, validation_data=(X_testNormalised, Y_test),  verbose=2, batch_size=30, epochs=500, callbacks=[EarlyStoppingCallback])

        return model, history


def trainTreeClassifier(classifier):
    print(f"\n- Training {classifier} ... \n")
    classifier.fit(X_train, Y_train, sample_weight=W_train)
    predictions = classifier.predict(X_test)
    confusionMatrix = confusion_matrix(y_true=Y_test, y_pred=predictions)
    print("Confusion Matrix \n", confusionMatrix, "\n")
    print("Score:", classifier.score(X_test, Y_test), "\n")

    return classifier


def createSingleROCcurve(classifier):
    Y_pred = classifier.predict_proba(X_test)[:, 1]
    falsePositiveRate, truePositiveRate, threshold = roc_curve(Y_test, Y_pred)
    aucScore = roc_auc_score(Y_test, Y_pred)

    return falsePositiveRate, truePositiveRate, aucScore


def trainAll():

    #TODO Simple BDT classifier
    #compare to lais's analysis 
    #experiment with other training variables
    #Compare to ROOT analysis
    #take notes of ALL experiments
    #learn manim (optional)


    RFClassifier = RandomForestClassifier(n_estimators=200)
    RFClassifier = trainTreeClassifier(RFClassifier)
    RFfalsePositiveRate, RFtruePositiveRate, RFaucScore = createSingleROCcurve(RFClassifier)
    print(f"\n RF AUC Score: {RFaucScore}")

    GBClassifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
    GBClassifier = trainTreeClassifier(GBClassifier)
    GBfalsePositiveRate, GBtruePositiveRate, GBaucScore = createSingleROCcurve(GBClassifier)
    print(f"\n GB AUC Score: {GBaucScore}")

    ADClassifier = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, algorithm="SAMME")
    ADClassifier = trainTreeClassifier(ADClassifier)
    ADfalsePositiveRate, ADtruePositiveRate, ADaucScore = createSingleROCcurve(ADClassifier)
    print(f"\n AD AUC Score: {ADaucScore}")

    NNmodel, NNhistory = CreateNNModel()
    Y_pred = NNmodel.predict(X_testNormalised)
    NNfalsePositiveRate, NNtruePositiveRate, threshold = roc_curve(Y_test, Y_pred)
    NNaucScore = roc_auc_score(Y_test, Y_pred)
    confusionMatrixNN = confusion_matrix(y_true=Y_test, y_pred=np.round(Y_pred))
    print("Confusion Matrix \n", confusionMatrixNN, "\n")
    print(f"\n NN AUC Score: {NNaucScore}")
    
    plt.figure(figsize=(10, 8))
    plt.plot(NNfalsePositiveRate, NNtruePositiveRate, label="Neural Network")
    plt.plot(RFfalsePositiveRate, RFtruePositiveRate, label="Random Forest")
    plt.plot(GBfalsePositiveRate, GBtruePositiveRate, label="Gradient Boosting")
    plt.plot(ADfalsePositiveRate, ADtruePositiveRate, label="ADA Boosting")
    plt.plot([0, 1], [0, 1], "k--")  # Added diagonal line for random classifier
    plt.xlabel("False Positive Rate") 
    plt.ylabel("True Positive Rate")  
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{imagePath}SignalAndBackgroundROC.png")
    plt.close()
    print("\nAll Complete\n")


if __name__ == "__main__":
    trainAll()










