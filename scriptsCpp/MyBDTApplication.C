//Author: Lais Soares Lavra
//Last update: April 2024

#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include "TMath.h"

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TTreeFormula.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#include "TMVA/Factory.h"

using namespace std;


vector<TString> GetBDTVar(TString xml = ""){

  vector<TString> var;
  string temp;
  ifstream monFlux(xml);
  if(!monFlux)return var;

  //How many variables ?
  Int_t nVar=0;
  while(monFlux >> temp){
    if (temp.substr(0,10)=="Expression"){
          var.push_back(temp.substr(12,temp.size()-13));
    }
  }
  return var;
}


Int_t GetNSpect(TString xml = ""){

  Int_t nspec;
  string temp;
  ifstream monFlux(xml);
  if(!monFlux)return 0;

  //How many variables ?
  Int_t nVar=0;
  while(monFlux >> temp){

    if (temp.substr(0,5)=="NSpec"){
      nspec= stoi(temp.substr(7,temp.size()-9));
    }
  }
  return nspec;

}




void MyBDTApplication(TString year = "2018", TString gb="6"){

   // Create the Reader object
   TMVA::Tools::Instance();
   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

   // Create a set of variables and declare them to the reader
   // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
  TString BDTname= "BDT_"+year; 
  TString wPath="./dataset";
  //TString wPath="/users/divers/lhcb/kar/private/BDT/";

  TString xml_fold1 = wPath+"/weights/TMVAClassification_"+BDTname+"_fold1.weights.xml" ;
  TString xml_fold2 = wPath+"/weights/TMVAClassification_"+BDTname+"_fold2.weights.xml";
  //if (gb==0){


  ///////////////////////////////////////////
  //READER VARIABLES & SPECTATORS :        //
  ///////////////////////////////////////////

  vector<TString> var=GetBDTVar(xml_fold1);
  var[2]="log(B_SmallestDeltaChi2OneTrack)*(B_SmallestDeltaChi2OneTrack>0)+20*(B_SmallestDeltaChi2OneTrack<0)";
  vector<Float_t*>value_F;

  for (Int_t i=0; i< static_cast<Int_t>(var.size())-GetNSpect(xml_fold1); i++){
    value_F.push_back(new Float_t(0));
    cout << var[i] << " " << *(value_F[i]) << endl;
    if (i==2) { reader->AddVariable("log(B_SmallestDeltaChi2OneTrack)*(B_SmallestDeltaChi2OneTrack>0)+20*(B_SmallestDeltaChi2OneTrack<0)",value_F[2]);}
    else{
    reader->AddVariable(var[i],value_F[i]);}
  }

  for (Int_t i=static_cast<Int_t>(var.size())-GetNSpect(xml_fold1); i< static_cast<Int_t>(var.size()); i++){
    value_F.push_back(new Float_t);
    reader->AddSpectator(var[i],value_F[i]);
  }



   // Book method(s)
  TString BDTfold1= BDTname+ TString("_fold1");
  TString BDTfold2= BDTname+ TString("_fold2");
  reader->BookMVA(BDTfold1, xml_fold1 );
  reader->BookMVA(BDTfold2, xml_fold2);

   // Prepare input tree (this must be replaced by your data source)
   //The application stage takes the BDT trained using signal and background samples, and applies it to a sample with an unknown mixture of signal and background (i.e. real data).

   std::cout << "--- Selecting signal sample" << std::endl;
   
   //TFile *input= TFile::Open("/home/lais/Desktop/BDT/GB2/kpiG_MC_Bd2KstGamma_prefilter-BDTPresel_v1_"+year+"_noPIDsel-noTrig_wPID_wGB2copy.root","update");
   //TFile *input= TFile::Open("/home/lais/Desktop/BDT/GB"+gb+"/kpiG_MC_Bd2KstGamma_prefilter-BDTPresel_v1_"+year+"_noPIDsel-noTrig_wPID_wGB"+gb+".root");
   //TFile *input= TFile::Open("./pipiG_MC_Bd2RhoGamma_HighPt_prefilterBDT_2018_noPIDsel.root");
   TFile *input= TFile::Open("./DATA_TUPLES/kpiG_"+year+"_finalcuts.root");
   TTree* treeinput = (TTree*)input->Get("DecayTree");
   std::cout << "--- TMVAClassificationApp    : Using input file: " << input->GetName() << std::endl;
   
   //output file which will contain the BDT response of each fold2t:
   //TFile *output = TFile::Open("/home/lais/Desktop/BDT/GB"+gb+"/kpiG_MC_Bd2KstGamma_prefilter-BDTPresel_v1_"+year+"_noPIDsel-noTrig_wPID_wGB"+gb+"_wBDT.root","RECREATE");
   //TFile *output = TFile::Open("./pipiG_MC_Bd2RhoGamma_HighPt_output.root","RECREATE");
   TFile *output = TFile::Open("./BDT_TUPLES/kpiG_"+year+"_finalcuts_output.root","RECREATE");
   TTree* theTree = treeinput->CopyTree("");
  // TTree* theTree = treeinput->CloneTree(0);


   //You might need to define some variables to add to the output file to view later

   Double_t random;
   theTree->SetBranchAddress("rndbit",&random);
   
//   if (theTree->GetBranch("selBDT") != NULL){

//    cout << "TBranch "<<"selBDT"<<" already exist, but branch actualisation is impossible with root ! please change the name of your BDT (name.xml)"<< endl;
//    return theTree;
//  }

 
   
   //Add a branch to store the BDT response of fold2ts:
   cout<<" fine at this moment"<<endl;
   double BDT;
   
   TBranch *BDT_branch = theTree->Branch("selBDTL",&BDT,"selBDTL/D");
   
   TStopwatch sw;
   sw.Start();

   std::cout << "Number of events that will be processed: " << theTree->GetEntries() << " events" << std::endl;
   
   vector<TTreeFormula*> form;

   for (Int_t iVar=0;iVar<static_cast<Int_t>(var.size());iVar++){
    form.push_back( new TTreeFormula(TString(var[iVar]),TString(var[iVar]), theTree));
    //TTreeFormula form(TString(var[iVar]),TString(var[iVar]), theTree);
  }
  
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++){
      if (ievt%100000 == 0) std::cout << "--- ... Processing event: " << ievt <<std::endl;
      theTree->LoadTree(ievt);
      theTree->GetEntry(ievt);

      for (Int_t iVar=0; iVar<static_cast<Int_t>(var.size());iVar++){
        *value_F[iVar]=(float)form[iVar]->EvalInstance();
        std::cout << ievt << " => " << iVar << " : " << *(value_F[iVar]) << endl;
      }
      
      
      double BDT1=reader->EvaluateMVA(BDTfold1);
      double BDT2=reader->EvaluateMVA(BDTfold2);
      std::cout << random << " =>  BDTs = " << BDT1 << " | " << BDT2 << endl;
      if (random>0.5){
        BDT=BDT1;
      }
      else{
        BDT=BDT2;
      }
      //return;
      BDT_branch->Fill();
     
      if (ievt%1000 == 0) std::cout << "--- ... Processing event: " << random << " "<<ievt << "  BDT = " <<BDT << std::endl;
     }
     
  sw.Stop();
  
  output->cd();
  theTree->Write();

  
  std::cout << "--- End of event loop: ";
  sw.Print();
  std::cout << "==> TMVAClassificationApplication is done for "<<BDTname<<" !" << std::endl << std::endl;
  
  output->Close();
  
  

}


