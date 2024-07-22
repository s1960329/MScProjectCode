//Author: Lais Soares Lavra
//Last update: April 2024

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"



void BuildBDT(TString gbflag = "6",TString year="2012", TString part="fold2" ){

//TString BDTname ="BDTGB"+gbflag+"_"+year+"_"+part ;
TString BDTname ="BDT_"+year+"_"+part ;
// Create a ROOT output file where TMVA will store ntuples, histograms, etc.
//TString outfileName("/home/lais/Desktop/BDT/BDT_v1/nfolding_results/BDT_min_noP_res_"+year+"_GB"+gbflag+"_"+part+".root");
TString outfileName("./nfolding_results/BDT_"+year+"_"+part+".root");

TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
  
// Create the factory object. Later you can choose the methods whose performance you'd like to investigate.
TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,"!Silent:Color:DrawProgressBar:AnalysisType=Classification" );
 
TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");

TString part1,part2,res;
part1="Kplus";
part2="piminus";
res="Kst_892_0";

dataloader->AddVariable("log(B_MINIPCHI2)");
dataloader->AddVariable("log(B_P)");
//dataloader->AddVariable("log(log(B_SmallestDeltaChi2OneTrack)*(B_SmallestDeltaChi2OneTrack>0)+1000*(B_SmallestDeltaChi2OneTrack<0))");
dataloader->AddVariable("log(B_SmallestDeltaChi2OneTrack)*(B_SmallestDeltaChi2OneTrack>0)+20*(B_SmallestDeltaChi2OneTrack<0)");
dataloader->AddVariable("log(B_FD_OWNPV)");
dataloader->AddVariable("-log(tan(atan(B_PT/B_PZ)/2))");
dataloader->AddVariable("log(min(piminus_IP_OWNPV,Kplus_IP_OWNPV))");
dataloader->AddVariable("log("+part1+"_PT)");
dataloader->AddVariable("log("+part2+"_PT)");
dataloader->AddVariable("log("+res+"_IP_OWNPV)");

dataloader->AddVariable("log("+res+"_PT)");


if (year=="2015"|| year=="2016" || year=="2017"|| year=="2018"){
    cout << "Variable cone added" << endl;
    dataloader->AddVariable("B_Cone3_B_ptasy");
  }

if (year=="2011" || year=="2012"){
    cout << "Isolation variable not added" << endl;
  } 



    //Spectators :
dataloader->AddSpectator("B_MM");
dataloader->AddSpectator(res+"_MM");
dataloader->AddSpectator(res+"_thetaH");
 
//Specifying training and test data 


 //Signal 
TChain* treeSignal = new TChain("DecayTree");
if (gbflag=="6"){
	treeSignal->Add("./MC_TUPLES/MC_kpiG_"+year+"_trueid.root");
}

if (gbflag=="2"){
	treeSignal->Add("/home/lais/Desktop/BDT/GB"+gbflag+"/kpiG_"+year+"_trueid.root");
}
cout << "KstGamma" << endl;
cout << "YEAR: " << year << endl;

cout << "signal_fold2 : cuts applied (rndbit<0.5)";
TTree *signal_fold2 = treeSignal->CopyTree("rndbit<0.5");
cout << "done with "<< signal_fold2->GetEntries()  <<" entries " <<endl; 

cout << "signal_fold1 : cuts applied (rndbit>0.5)";
TTree *signal_fold1 = treeSignal->CopyTree("rndbit>0.5");
cout << "done with "<< signal_fold1->GetEntries()  <<" entries !" <<endl; 


//TTree *signal_fold2 = treeSignal->CopyTree("(int(random*1000000000)%2!=0)");
//TTree *signal_fold2 = treeSignal->CopyTree("eventNumber%2!=0");
//TTree *signal_fold2 = treeSignal->CopyTree("(eventNumber%100<50)");

//TTree *signal_fold1 = treeSignal->CopyTree("(int(random*1000000000)%2==0)");
//TTree *signal_fold1 = treeSignal->CopyTree("(eventNumber%2==0)");
//TTree *signal_fold1 = treeSignal->CopyTree("(eventNumber%100>49)");


//Background
TChain* treeBkg = new TChain("DecayTree");
treeBkg->Add("./DATA_TUPLES/data_kpiG_sideband_"+year+".root"); 
cout << "bkg_fold2 : cuts applied (rndbit<0.5)";

TTree *background_fold2 = treeBkg->CopyTree("rndbit<0.5");
cout << "done with "<< background_fold2->GetEntries()  <<" entries !" <<endl;

cout << "bkg_fold1  : cuts applied (rndbit>0.5)";
TTree *background_fold1 = treeBkg->CopyTree("rndbit>0.5");
cout << "done with "<< background_fold1->GetEntries()  <<" entries !" <<endl;
 

std::cout << "--- TMVAClassification : Using input signal file: " << treeSignal->GetName() << std::endl;
std::cout << "--- TMVAClassification : Using input background file: " << treeBkg->GetName() << std::endl;

//--- Register the training and test trees
//fold2
if (part=="fold2"){
    dataloader->AddSignalTree(signal_fold2,1,"Training");
    dataloader->AddSignalTree(signal_fold1,1,"Test");
    dataloader->AddBackgroundTree(background_fold2,1,"Training");
    dataloader->AddBackgroundTree(background_fold1,1,"Test");
  }
  //fold1
if (part == "fold1"){
    dataloader->AddSignalTree(signal_fold1,1,"Training");
    dataloader->AddSignalTree(signal_fold2,1,"Test");
    dataloader->AddBackgroundTree(background_fold1,1,"Training");
    dataloader->AddBackgroundTree(background_fold2,1,"Test");
  }



if (gbflag=="6"){
    //std::cout << "--- Weights applied : GB0 only " << std::endl;
    //dataloader->SetSignalWeightExpression("gb_weights");
    std::cout << "--- Weights applied : PID+GB6 " << std::endl;
    dataloader->SetSignalWeightExpression("gb6_weights*Event_PIDCalibEff*IsPhoton_weight");
    
  }
/*  //fold1
if (gbflag=="2"){
    //std::cout << "--- Weights applied : PID+GB2 " << std::endl;
    //dataloader->SetSignalWeightExpression("gb2_weights");
    std::cout << "--- Weights applied : PID+GB2 " << std::endl;
    dataloader->SetSignalWeightExpression("gb2_weights*Event_PIDCalibEff*IsPhoton_weight");
  }  */
 


TCut cutSignal="";
TCut cutBKG="";
 
//dataloader->PrepareTrainingAndTestTree(cutSignal,cutBKG,"");

dataloader->PrepareTrainingAndTestTree("", "");


//factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"!H:NTrees=800:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20" );

if (year=="2011" ){ 
	cout << "Book option Run 1 : NTrees=200:MaxDepth=3:MinNodeSize=5%" << endl;
	factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"H:NTrees=200:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");
	factory->BookMethod(dataloader,TMVA::Types::kBDT,"test1","H:NTrees=250:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");
	factory->BookMethod(dataloader,TMVA::Types::kBDT,"test2","H:NTrees=300:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");
}

if (year=="2012" ){ 
	cout << "Book option Run 1 : NTrees=400:MaxDepth=3:MinNodeSize=2.5%" << endl;
	factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"H:NTrees=400:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=2.5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");


}


if (year=="2015"){ 

	cout << "Book option Run 2 : NTrees=500:MaxDepth=3:MinNodeSize=2.5%" << endl;
factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"H:NTrees=500:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=2.5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");


}


if (year=="2016"){ 

	cout << "Book option Run 2 : NTrees=600:MaxDepth=3:MinNodeSize=2.5%" << endl;
factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"H:NTrees=600:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=2.5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");

}


if (year=="2017"){ 

	cout << "Book option Run 2 : NTrees=800:MaxDepth=3:MinNodeSize=2.5%" << endl;
	factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"H:NTrees=800:MaxDepth=3:IgnoreNegWeightsInTraining:MinNodeSize=2.5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");

}



if (year=="2018"){ 

	cout << "Book option Run 2 : NTrees=800:MaxDepth=4:MinNodeSize=2.5%" << endl;
	factory->BookMethod(dataloader, TMVA::Types::kBDT,BDTname,"H:NTrees=800:MaxDepth=4:IgnoreNegWeightsInTraining:MinNodeSize=2.5%:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.6:SeparationType=GiniIndex:nCuts=20");

}


factory->TrainAllMethods();
factory->TestAllMethods();
factory->EvaluateAllMethods();

outputFile->Close();

std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
std::cout << "==> TMVAClassification is done!" << std::endl;

delete factory;
delete dataloader;
 
if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

return;

}

void MyTMVAClassification_nfold(TString gbflag = "6",TString year="2018", TString part="fold1" ){
     BuildBDT(gbflag,year,"fold1");
     BuildBDT(gbflag,year,"fold2");
}


 

 
   
