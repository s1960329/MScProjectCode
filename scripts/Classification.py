import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.metrics            import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection    import train_test_split
from misc                       import fileNamesCSV, uniqueVariables
from sklearn.ensemble           import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


trainingVariables = ["nTracks","B_P","gamma_PT","daughter_neutral_PT","B_Cone3_B_ptasy","B_ETA"]



def formatForClassification(normalise=False):
    
    signalDict = dict(zip(uniqueVariables["pipi"],uniqueVariables["any"]))
    signalData = pd.read_csv(fileNamesCSV["pipi"], index_col=0)
    signalData = signalData.rename(columns = signalDict)
    signalData["isSignal"] = np.ones(len(signalData))

    backgroundDict = dict(zip(uniqueVariables["kpi"],uniqueVariables["any"]))
    backgroundData = pd.read_csv(fileNamesCSV["kpisb"], index_col=0)
    backgroundData = backgroundData.rename(columns = backgroundDict)
    backgroundData["weights"]  = np.ones(len(backgroundData))
    backgroundData["isSignal"] = np.zeros(len(backgroundData))

    FullData = pd.concat([backgroundData, signalData])

    train, test = train_test_split(FullData)

    X_train = train[trainingVariables]
    Y_train = train["isSignal"]
    W_train = train["weights"]

    X_test  = test[trainingVariables]
    Y_test  = test["isSignal"]
    W_test  = test["weights"]

    return ((X_train,Y_train,W_train),(X_test,Y_test,W_test))


def normaliseData(data):
    for var in trainingVariables:
        data[var] = (data[var] - min(data[var])) / (max(data[var])  -  min(data[var]) )

    return data
    

((X_train,Y_train,W_train),(X_test,Y_test,W_test)) = formatForClassification()
X_trainNormalised = normaliseData(X_train)
X_testNormalised  = normaliseData(X_test)

def CreateNNModel():
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
    
    plt.plot(NNfalsePositiveRate, NNtruePositiveRate, label="Neural Network")
    plt.plot(RFfalsePositiveRate, RFtruePositiveRate, label="Random Forest")
    plt.plot(GBfalsePositiveRate, GBtruePositiveRate, label="Gradient Boosting")
    plt.plot(ADfalsePositiveRate, ADtruePositiveRate, label="ADA Boosting")
    plt.plot([0, 1], [0, 1], "k--")  # Added diagonal line for random classifier
    plt.xlabel("False Positive Rate")  # Added x-axis label
    plt.ylabel("True Positive Rate")  # Added y-axis label
    plt.title("ROC Curve")  # Added title'
    plt.legend()
    plt.show()



if __name__ == "__main__":
    trainAll()










