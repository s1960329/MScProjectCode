from misc import *
from sklearn.model_selection import train_test_split

def combineSignalAndBackground(mode = "pipi"):

    BackgrData = pd.read_csv(f"data/{mode}/BackgroundData.csv", index_col = 0)
    SignalData = pd.read_csv(f"data/{mode}/SignalData.csv",     index_col = 0)

    VarDict    = dict(zip(uniqueFeatures[mode],uniqueFeatures["any"]))
    BackgrData = BackgrData.rename(columns = VarDict)
    SignalData = SignalData.rename(columns = VarDict)

    BackgrData["weights"]  = np.ones(len(BackgrData))

    BackgrData["isSignal"] = np.zeros(len(BackgrData))
    SignalData["isSignal"] = np.ones( len(SignalData))

    FullData = pd.concat([BackgrData, SignalData])
    FullData = FullData.sample(frac=1)
    FullData = FullData.reset_index(drop=True)
    
    FullData.to_csv(f"data/{mode}/FullData.csv")

def createTestAndTrainData(mode = "pipi"):

    FullData = pd.read_csv(f"data/{mode}/FullData.csv", index_col = 0)
    train, test = train_test_split(FullData)

    test. to_csv(f"data/{mode}/TestData.csv")
    train.to_csv(f"data/{mode}/TrainData.csv")
   
def splitSignalEvenly(filePath):
    Data = pd.read_csv(filePath, index_col=0)

    SampleSize = min( len(Data[Data["isSignal"] == 0]), len(Data[Data["isSignal"] == 1]) )
    EvenData = Data.groupby("isSignal", as_index=False, group_keys=False).apply(lambda x: x.sample(SampleSize))
    EvenData = EvenData.sample(frac=1)
    
    return EvenData

def createEvenTestAndTrainData(mode = "pipi"):
    
    testEven  = splitSignalEvenly(f"data/{mode}/TestData.csv")
    testEven. to_csv(f"data/{mode}/EvenTestData.csv")
    
    trainEven = splitSignalEvenly(f"data/{mode}/TrainData.csv")
    trainEven.to_csv(f"data/{mode}/EvenTrainData.csv")

def createSampleData():


    csvPath = "/Users/finnjohnonori/Documents/GitHubRepositories/MScProject/MScProjectCode/data/SampleData.csv"
    SampleData = readRootFile(fileNamesRoot["sm"], allFeatures["kpi"] + ["NB0_Kpigamma_sw"])

    renameDict = dict(zip(allFeatures["kpi"] + ["NB0_Kpigamma_sw"], allFeatures["any"] + ["weights"]))
    SampleData = SampleData.rename(columns=renameDict)
    SampleData.to_csv(csvPath)

if __name__ == "__main__":
    createSampleData()