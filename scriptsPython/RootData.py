import ROOT
from misc  import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import confusion_matrix, roc_curve, roc_auc_score



def ROOTplotROCcurve():

    BDT2018name_fold1 = f"{dataPath}/dataROOT/BDT_2018_fold1.root"
    BDT2018name_fold2 = f"{dataPath}/dataROOT/BDT_2018_fold2.root"

    BDT2018_fold1 = ROOT.TFile.Open(BDT2018name_fold1,"READ")
    BDT2018_fold2 = ROOT.TFile.Open(BDT2018name_fold2,"READ")

    histoBDT2018_fold1 = BDT2018_fold1.Get(f"dataset/Method_BDT/BDT_2018_fold1/MVA_BDT_2018_fold1_rejBvsS")
    histoBDT2018_fold2 = BDT2018_fold2.Get(f"dataset/Method_BDT/BDT_2018_fold2/MVA_BDT_2018_fold2_rejBvsS")

    legend = ROOT.TLegend(0.2, 0.4, 0.35, 0.58)
    legend.AddEntry(histoBDT2018_fold1,"2018 fold1 B^{0}#rightarrow K^{*0}#gamma")
    legend.AddEntry(histoBDT2018_fold2,"2018 fold2 B^{0}#rightarrow K^{*0}#gamma")

    canvas = ROOT.TCanvas("c1", "", 600, 350)
    canvas.SetGrid()

    histoBDT2018_fold1.SetLineColor(4)
    histoBDT2018_fold1.SetLineWidth(2)
    histoBDT2018_fold1.Draw()

    histoBDT2018_fold2.SetLineColor(5)
    histoBDT2018_fold2.SetLineWidth(2)
    histoBDT2018_fold2.Draw("SAME")

    legend.Draw()
    canvas.SaveAs(f"imgs/ROC_curve_2018_kpiG_both.png")



def ROOTloadData(trainDataPath = fileNamesCSV["f1t"], testDataPath = fileNamesCSV["f1v"]):
    train     = pd.read_csv(trainDataPath, index_col=0)
    test      = pd.read_csv(testDataPath, index_col=0)
    variables = list(train.keys())[ 2 : len(list(train.keys())) - 2 ]

    X_test  =  test[variables]
    Y_test  =  test["classID"]
    W_test  =  test["weight" ]

    X_train = train[variables]
    Y_train = train["classID"]
    W_train = train["weight" ]

    print("\n Data loaded...")

    return ((X_train,Y_train,W_train),(X_test,Y_test,W_test))



def ROOTcreateBDT(trainDataPath = fileNamesCSV["f1t"] , testDataPath = fileNamesCSV["f1v"]):
    print("\n Creating BDT...")
    ((X_train,Y_train,W_train),(X_test,Y_test,W_test)) = ROOTloadData(trainDataPath, testDataPath)
    BDT = GradientBoostingClassifier( n_estimators=800, learning_rate=0.00625, max_depth=5)

    print("\n", BDT)
    print("\n Training BDT...")
    BDT.fit(X_train, Y_train, sample_weight=W_train)
    cm = confusion_matrix(y_pred=BDT.predict(X_test), y_true=Y_test)

    print("\n Predictions...")
    print("\n",  cm)
    print("\n", (cm[1,1] + cm[0,0]) / len(Y_test) )
    print("\n Complete!")
    return BDT



if __name__ == "__main__":
    ROOTplotROCcurve()

