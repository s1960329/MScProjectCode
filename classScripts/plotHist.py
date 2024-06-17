from typing import Any 
import csv
import math
import uproot
#import hep_ml

import ROOT

class PlotCreator():

    def __init__(self) -> None:

        ROOT.gROOT.SetBatch(True)
        ROOT.gStyle.SetGridColor(17)

        self.CANVAS_SIZE_X   = 1000
        self.CANVAS_SIZE_Y   = 800
        self.N_BINS          = 50
        self.LEGEND_X        = 0.73
        self.LEGEND_Y        = 0.90
        self.LEGEND_SIZE_X   = 0.1
        self.LEGEND_SIZE_Y   = 0.07
        self.HIST_LINE_WIDTH = 2
        self.FONT            = 43
        self.AXIS_TITLE_SIZE = 20
        self.LABEL_SIZE      = 20
        self.MARKER_STYLE    = 1
        self.MARKER_SIZE     = 20
        self.UPPER_BOUND_Y   = 1
        self.LOWER_BOUND_Y   = 0
        self.UPPER_BOUND_X   = 1
        self.LOWER_BOUND_X   = 0
        
        self.COLOR           = {"kpi"  : ROOT.kBlue + 2,
                                "pipi" : ROOT.kOrange,
                                "kpisw": ROOT.kRed + 1,
                                "ratio": ROOT.kBlack}
        
        self.sweights   = "(abs(B_M01-895.55)<100)*NB0_Kpigamma_sw"
        self.trueid     = "(B_BKGCAT==0 ||B_BKGCAT== 50)"
        self.decays     = ["kpi", "pipi", "kpisw"]


        self.loadTrees()
        self.getVariableNames()

    def loadTrees(self):
        # Uses sweights to remove background from sample data
        self.Histograms = {}
        self.Trees      = {}
        self.Sizes      = {}

        # imports data from the .root files that contain Monte Carlo data
        # imports data from the Kaon decays and adds them to a TTree object
        Tree_kpi = ROOT.TChain("DecayTree")
        Tree_kpi.Add("../data/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        Tree_kpi.Add("../data/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root")

        Tree_pipi = ROOT.TChain("DecayTree")
        Tree_pipi.Add("../data/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        Tree_pipi.Add("../data/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        
        # imports data from the Sample decays and adds them to a TTree object
        Tree_kpisw = ROOT.TChain("DecayTree")
        Tree_kpisw.Add("../data/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root")
        
        # stores trees
        self.Trees["kpi"  ] = Tree_kpi
        self.Trees["kpisw"] = Tree_kpisw
        self.Trees["pipi" ] = Tree_pipi

        # defines the lengths of the TTree objects
        self.Sizes["kpi"  ] = self.Trees["kpi"  ].GetEntries()
        self.Sizes["kpisw"] = self.Trees["kpisw"].GetEntries()
        self.Sizes["pipi" ] = self.Trees["pipi" ].GetEntries()

    def getVariableNames(self):
        #Imports the variable names and finds which variables belong to each decay
        #Import all the Kaon decay variables
        variables_kpi = list(self.Trees["kpi"].GetListOfBranches()) 
        variablesNames_kpi = [ str(var.GetFullName()) for var in variables_kpi]

        #Imports all the Sample decay variables
        variables_kpisw = list(self.Trees["kpisw"].GetListOfBranches()) 
        variablesNames_kpisw = [ str(var.GetFullName()) for var in variables_kpisw]

        variables_pipi = list(self.Trees["pipi"].GetListOfBranches()) 
        variablesNames_pipi = [ str(var.GetFullName()) for var in variables_pipi]
        
        variablesNames_kpi.sort()
        variablesNames_kpisw.sort()
        variablesNames_pipi.sort()

        self.CommonVariables = list( set(variablesNames_kpisw) & set(variablesNames_kpi) )
        self.kpiVariables    = list( set(variablesNames_kpi) - set(variablesNames_kpisw) )
        self.kpiswVariables  = list( set(variablesNames_kpisw) - set(variablesNames_kpi) )
        self.AllVariables    = self.CommonVariables + self.kpiVariables + self.kpiswVariables 

        #sorts lists before saving
        self.CommonVariables.sort()
        self.kpiVariables   .sort()
        self.kpiswVariables .sort()
        self.AllVariables   .sort()
        
    def findVariableBounds(self, variable):
        #Finds the maximium and minimum values for each variable
        #Creates a list of possible bounds
        #Save the top and bottom entries of the sorted list
        self.LOWER_BOUND_X = float(max(self.Trees["kpisw"].GetMinimum(variable), self.Trees["kpi"  ].GetMinimum(variable)))
        self.UPPER_BOUND_X = float(min(self.Trees["kpi"  ].GetMaximum(variable), self.Trees["kpisw"].GetMaximum(variable)))


    def createHist(self, variable, decay = "kpi"):
        #Computes variable bounds
        self.findVariableBounds(variable)
        
        #Defines variable histogram
        hist_name = f"h_{variable}_{decay}"
        hist = ROOT.TH1F(hist_name, "",  self.N_BINS, self.LOWER_BOUND_X, self.UPPER_BOUND_X)

        hist.SetMarkerStyle(self.MARKER_STYLE)
        hist.SetMarkerSize(self.MARKER_SIZE)

        hist.SetLineColor(self.COLOR[decay])
        hist.SetLineWidth(self.HIST_LINE_WIDTH)
       
        #Draw with sWeights if the sample data is used
        if decay == "kpisw": self.Trees[decay].Draw(f"{variable}>>{hist_name}", self.sweights)
        else:                self.Trees[decay].Draw(f"{variable}>>{hist_name}")
        
        #Saves updated sizes once sweights are applied
        self.Sizes[decay]      = hist.GetEntries()
        self.Histograms[decay] = hist
    
    def createSingleImage(self, variable, decay = "kpi"):
        #Defines single histogram canvas
        self.canvas = ROOT.TCanvas(f"c_{variable}_{decay}", f"{variable}_{decay}", self.CANVAS_SIZE_X, self.CANVAS_SIZE_Y)
        
        #Defines the histogram
        self.createHist(variable, decay)
        self.Histograms[decay].GetYaxis().SetRangeUser(0, 1.1*self.Histograms[decay].GetBinContent(self.Histograms[decay].GetMaximumBin()))
        self.Histograms[decay].DrawNormalized("HISTO", norm=1)
        
        h_singlex = self.Histograms[decay].GetXaxis()
        h_singlex.SetLabelFont(self.FONT)
        h_singlex.SetLabelSize(self.LABEL_SIZE)
        
        #Draws and saves the full image
        self.canvas.SetGridx()
        self.canvas.Draw()
        self.canvas.Print(f"../imgs/plot_{variable}_{decay}_single.pdf")
        self.canvas.Close()

    def createAllSingleImages(self):
        #creates a single histogram for each variable and decay types
        for variable in self.kpiVariables:
            self.createSingleImage(variable, decay="kpi")
            self.createSingleImage(variable, decay="kpisw")
        

    def createRatioPlot(self, variable):
        #Defines and normalises Ratio plot
        h_ratio = self.Histograms["kpisw"].Clone("h_ratio")
        h_ratio.Divide(self.Histograms["kpi"])
        
        integral_kpi   = self.Histograms["kpi"  ].Integral()
        integral_kpisw = self.Histograms["kpisw"].Integral()
        h_ratio.Scale(integral_kpi/integral_kpisw)

        # Style for Ratio plot
        h_ratio.SetTitle("")

        h_ratio.SetLineColor(17)
        h_ratio.SetMarkerStyle(self.MARKER_STYLE)
        h_ratio.SetMarkerStyle(self.MARKER_STYLE)
        h_ratio.SetFillColorAlpha(ROOT.kBlack,0.5)
        
        self.UPPER_BOUND_ratio_Y = h_ratio.GetMaximum()
        self.LOWER_BOUND_ratio_Y = h_ratio.GetMinimum()
        
        # Style for y axis of Ratio plots
        h_ratioy = h_ratio.GetYaxis()
        h_ratioy.SetRangeUser(self.LOWER_BOUND_ratio_Y, self.UPPER_BOUND_ratio_Y)
        h_ratioy.SetTitle("Ratio")
        h_ratioy.SetTitleOffset(2.0)
        h_ratioy.SetTitleSize(self.AXIS_TITLE_SIZE)
        h_ratioy.SetTitleFont(self.FONT)
        h_ratioy.SetLabelFont(self.FONT)
        h_ratioy.SetLabelSize(self.LABEL_SIZE)
        h_ratioy.CenterTitle()

        # Style for x axis of Ratio plot
        h_ratiox = h_ratio.GetXaxis()
        h_ratiox.SetTitle(variable)
        h_ratiox.SetTitleOffset(1.2)
        h_ratiox.SetTitleSize(0.095)
        h_ratiox.SetLabelFont(self.FONT)
        h_ratiox.SetLabelSize(self.LABEL_SIZE)
        h_ratiox.CenterTitle()
        
        h_ratio.Draw("E4")
        
        
        #Saves histogram 
        self.Histograms["ratio"] = h_ratio
    
    def createLegend(self):
        legend = ROOT.TLegend(self.LEGEND_X, self.LEGEND_Y, self.LEGEND_X + self.LEGEND_SIZE_X, self.LEGEND_Y + self.LEGEND_SIZE_Y)
        legend.AddEntry(self.Histograms["kpi"  ], "B^{0}#rightarrow K^{*0}#gamma - MC")
        legend.AddEntry(self.Histograms["kpisw"], "B^{0}#rightarrow K^{*0}#gamma - Sample")
        legend.SetBorderSize(0)
        legend.SetTextSize(0.034)

        self.legend = legend

    def createDoubleCanvas(self, variable):
        canvas = ROOT.TCanvas(f"c_{variable}", f"canvas_{variable}", self.CANVAS_SIZE_X, self.CANVAS_SIZE_Y)

        padHist = ROOT.TPad("padHist", "padHist", 0, 0.3, 1, 1.0)
        padHist.SetGridx()
        padHist.SetTopMargin(0.12)
        padHist.SetBottomMargin(0.025)
        padHist.Draw()

        canvas.cd()
        padRatio = ROOT.TPad("padRatio", "padRatio", 0, 0.01, 1, 0.3)

        padRatio.SetGridx()
        padRatio.SetTopMargin(0.025)
        padRatio.SetBottomMargin(0.28)
        padRatio.Draw()

        self.canvas   = canvas
        self.padHist  = padHist
        self.padRatio = padRatio

    def createDoubleImage(self, variable):
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetTitleAlign(33)
        ROOT.gStyle.SetTitleX(.63)
        ROOT.gStyle.SetTitleY(.97)

        
        self.createHist(variable, "kpisw")
        self.createHist(variable, "kpi")
        self.createDoubleCanvas(variable)
        self.padHist.cd()

        self.kpi_integral   = not (self.Histograms["kpisw"].Integral() == 0)
        self.kpisw_integral = not (self.Histograms["kpi"  ].Integral() == 0)
        
        if self.kpi_integral and self.kpi_integral:

            self.Histograms["kpi"  ].Scale(1/self.Histograms["kpi"  ].Integral())
            self.Histograms["kpisw"].Scale(1/self.Histograms["kpisw"].Integral())

            self.Histograms["kpi"].GetXaxis().SetTitle("Variable")
            self.Histograms["kpi"].SetTitle("Ratio Distribution")

            yb_kpisw = self.Histograms['kpisw'].GetBinContent(self.Histograms['kpisw'].GetMaximumBin())
            yb_kpi   = self.Histograms['kpi'  ].GetBinContent(self.Histograms['kpi'  ].GetMaximumBin())
            
            self.LOWER_BOUND_hist_Y = 0
            self.UPPER_BOUND_hist_Y = float('%.*g' % (2, max(yb_kpi, yb_kpisw)*1.2))

            self.Histograms["kpi"].GetYaxis().SetRangeUser(self.LOWER_BOUND_hist_Y, self.UPPER_BOUND_hist_Y)
            self.Histograms["kpi"].GetXaxis().SetLabelSize(0.0)
            self.Histograms["kpi"  ].Draw("HISTO")
            self.Histograms["kpisw"].Draw("HISTO SAME")

            self.createLegend()
            self.legend.Draw("SAME")

            self.padRatio.cd()
            self.createRatioPlot(variable)

            self.Histograms["kpi"  ].GetXaxis().SetRangeUser(self.LOWER_BOUND_X, self.UPPER_BOUND_X)
            self.Histograms["kpisw"].GetXaxis().SetRangeUser(self.LOWER_BOUND_X, self.UPPER_BOUND_X)
            self.Histograms["kpi"  ].Smooth()
            self.Histograms["kpisw"].Smooth()

            self.Histograms["ratio"].Smooth()
            self.Histograms["ratio"].Draw("E4")

            #Draws and saves the full image
            self.canvas.Draw()
            self.canvas.Print(f"../imgs/plot_{variable}.pdf")
            self.canvas.Close()

    def createAllDoubleImages(self):
        for variable in self.CommonVariables:
            self.createDoubleImage(variable)

    def computeDifference(self, variable):
        #Returns the P value
        self.createHist(variable, "kpisw")
        self.createHist(variable, "kpi"  )
        return self.Histograms["kpi"].Chi2Test(self.Histograms["kpisw"])
    
    def computeAllDifferences(self):
        diffDict = {}
        for var in self.CommonVariables:
            diff = self.computeDifference(var)
            diffDict[var] = diff
        
        file = open("../data/Chi2Values.csv", "wb")
        write = csv.DictWriter(file, diffDict.keys())
        write.writerows(diffDict)
        file.close()


if __name__ == "__main__":
    p = PlotCreator()

    # p.createSingleImage("gamma_PT")
    # * log(B_MINIPCHI2)
    # * log(B_FD_OWNPV)
    # * B_Cone3_B_ptasy
    # * gamma_PT
    # * log(piminus_IP_OWNPV)
    # * piminus_PT
    p.createDoubleImage("gamma_PT")
    p.createDoubleImage("piminus_PT")
    p.createDoubleImage("piminus_IP_OWNPV")
    p.createDoubleImage("B_Cone3_B_ptasy")
    p.createDoubleImage("B_FD_OWNPV")
    p.createDoubleImage("B_MINIPCHI2")

    p.createDoubleImage("Kplus_IP_OWNPV")
    p.createDoubleImage("Kplus_PT")
    p.createDoubleImage("Kst_892_0_PT")
    p.createDoubleImage("Kst_892_0_IP_OWNPV")

    
    #p.createAllDoubleImages()

    #B_BMassFit_Kst_892_0_Kplus_PX
    #B_Cone2_B_pt
    #gamma_PT
    #B_M
