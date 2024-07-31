import joblib
from misc import histStyle
from scriptsPython.Classifiers import *
from sklearn.metrics import ConfusionMatrixDisplay


def loadModels(name):
    GB = joblib.load(f"savedModels/{name}/GB{name}.joblib")
    RF = joblib.load(f"savedModels/{name}/RF{name}.joblib")
    AD = joblib.load(f"savedModels/{name}/AD{name}.joblib")
    NN = joblib.load(f"savedModels/{name}/NN{name}.joblib")

    return (GB,RF,AD,NN)


def createROCplot(name):

    fold = "fold2"
    BDT2018name = "../data/dataROOT/BDT_2018_"+str(fold)+".root"
    file = uproot.open(BDT2018name)
    ROClais = file["dataset/Method_BDT/BDT_2018_fold2/MVA_BDT_2018_fold2_rejBvsS"].values()
    ROClaisEdges = file["dataset/Method_BDT/BDT_2018_fold2/MVA_BDT_2018_fold2_rejBvsS"].axis().edges()

    (GB,RF,AD,NN) = loadModels(name)

    (GBfalsePositiveRate, GBtruePositiveRate) = GB.createROCcurve()
    (RFfalsePositiveRate, RFtruePositiveRate) = RF.createROCcurve()
    (ADfalsePositiveRate, ADtruePositiveRate) = AD.createROCcurve()
    (NNfalsePositiveRate, NNtruePositiveRate) = NN.createROCcurve(NN.X_test)

    fold = "fold2"
    BDT2018name = "../data/dataROOT/BDT_2018_"+str(fold)+".root"
    file = uproot.open(BDT2018name)
    ROClais = file["dataset/Method_BDT/BDT_2018_fold2/MVA_BDT_2018_fold2_rejBvsS"].values()
    ROClaisEdges = file["dataset/Method_BDT/BDT_2018_fold2/MVA_BDT_2018_fold2_rejBvsS"].axis().edges()

    plt.figure(figsize=(10, 8))
    plt.plot(NNfalsePositiveRate, NNtruePositiveRate,    alpha=0.7,    label="Neural Network")
    plt.plot(RFfalsePositiveRate, RFtruePositiveRate,    alpha=0.7,    label="Random Forest")
    plt.plot(GBfalsePositiveRate, GBtruePositiveRate,    alpha=0.7,    label="Gradient Boost")
    plt.plot(ADfalsePositiveRate, ADtruePositiveRate,    alpha=0.7,    label="Adaboost")
    plt.plot(1-ROClaisEdges[:-1], ROClais,               alpha=0.7,    label="ROOT Gradient Boost")
    plt.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.7)
    plt.xlabel("False Positive Rate") 
    plt.ylabel("True Positive Rate")  
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.grid(axis="both", linestyle="dashed", alpha=0.3)
    plt.show()


def createConfusionMatrices(name):
    (GB,RF,AD,NN) = loadModels(name)
    
    models     = [NN,GB,RF,AD]
    modelNames = ["Neural Network","Gradient Boosted","Random Forest", "Adaboost"]

    for model, modelName in zip(models, modelNames):
        cm = ConfusionMatrixDisplay(confusion_matrix(y_true=model.Y_test, y_pred=roundUp(model.predict(model.X_test).flatten())))
        cm.plot()
        plt.title(f"{name} {modelName} Confusion Matrix")
        plt.show()



if __name__ == "__main__":
    createROCplot(name="Bestpipi")
