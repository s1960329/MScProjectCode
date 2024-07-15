from numpy           import linspace
import matplotlib.pyplot as plt
from tensorflow      import keras
from misc            import formatForClassification, normaliseData,  FindBestCutOff, fileNamesCSV
from sklearn.metrics import confusion_matrix, roc_auc_score


isSignalEvenSplit = False
trainingVariables = ['nTracks', 'B_P', 'gamma_PT', 'daughter_neutral_PT', 'daughter_neutral_IP_OWNPV', 'daughterplus_PT', 'daughterplus_IP_OWNPV', 'B_Cone3_B_ptasy', 'B_ETA']
signalPath        = fileNamesCSV["pipi"]
backgroundPath    = fileNamesCSV["kpisb"]

((X_train,Y_train,W_train),(X_test,Y_test,W_test)) = formatForClassification(trainingVariables, signalPath, backgroundPath, isSignalEvenSplit)
X_trainNormalised = normaliseData(X_train)
X_testNormalised  = normaliseData(X_test)


FindBestCutOff(X_testNormalised, Y_test)




# NNw = NNroc/totalroc
# ADw = ADroc/totalroc
# GBw = GBroc/totalroc
# RFw = RFroc/totalroc

# NNroc = roc_auc_score(NNprediction, Y_test)
# ADroc = roc_auc_score(ADprediction, Y_test)
# GBroc = roc_auc_score(GBprediction, Y_test)
# RFroc = roc_auc_score(RFprediction, Y_test)

# totalroc = NNroc + ADroc + GBroc + RFroc