# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
import pandas as pd
# data sample
data = pd.read_csv("data/pipi/FullData.csv")
data = data[["nTracks","B_P","piminus_PT","isSignal", "weights"]]
data = array(data)

FoldedTrainData = []
FoldedTestData  = []

kfold = KFold(10)
# enumerate splits
for train, test in kfold.split(data):
    FoldedTrainingData.append(pd.DataFrame(data[train], columns=["nTracks","B_P","piminus_PT","isSignal", "weights"]))
    FoldedTestData.append(pd.DataFrame(data[test], columns=["nTracks","B_P","piminus_PT","isSignal", "weights"]))



print(FoldedTrainingData)