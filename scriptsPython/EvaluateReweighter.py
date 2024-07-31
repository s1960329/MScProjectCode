
from misc import SplitDataForTraining, normaliseData
from misc import fileNamesCSV

from tensorflow.keras.models    import Sequential, clone_model # type: ignore
from tensorflow.keras.layers    import Input, Dense, Dropout, BatchNormalization   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping           # type: ignore

EarlyStoppingCallback = EarlyStopping(monitor="val_loss", patience=10)
trainingVariables = ["nTracks", "B_P", "gamma_PT", "daughter_neutral_PT", "B_Cone3_B_ptasy", "B_ETA"]

((X_train,Y_train,W_train),(X_test,Y_test,W_test)) = SplitDataForTraining(trainingVariables, signalPath=fileNamesCSV["pipi"], backgroundPath=fileNamesCSV["sm"], evenSplit=True)
X_trainNormalised = normaliseData(X_train, varToNorm=trainingVariables)
X_testNormalised  = normaliseData(X_test,  varToNorm=trainingVariables)

hiddenLayers      = [BatchNormalization(center=False, scale=False, axis = 1),
                     Dense(len(trainingVariables),    input_shape = (len(trainingVariables),),    activation="relu"),
                     Dense(len(trainingVariables)*10, input_shape = (len(trainingVariables),),    activation="relu"),
                     Dense(len(trainingVariables),    input_shape = (len(trainingVariables)*10,), activation="relu"),
                     Dense(1,                         input_shape = (len(trainingVariables),),    activation="sigmoid")]


NNmodel = Sequential()
for layer in hiddenLayers:
    NNmodel.add(layer)

NNmodelWithoutWeights = clone_model(NNmodel)
bs = int(len(X_testNormalised) / 1000)

print("\n Training Classifier without weights ")
NNmodel.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])
history = NNmodel.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=bs, epochs=500, callbacks=[EarlyStoppingCallback])

print("\n Training Classifier with weights ")
NNmodelWithoutWeights.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"])
history = NNmodelWithoutWeights.fit(X_train, Y_train, sample_weight = W_train, validation_data=(X_test, Y_test), batch_size=bs, epochs=500, callbacks=[EarlyStoppingCallback])