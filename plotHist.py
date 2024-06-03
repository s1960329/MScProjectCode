from typing import Any
import ROOT
import csv

class PlotCreator():

    def __init__(self) -> None:
        self.CanvasSize  = 800
        self.NoOfBins    = 50
        self.legend_Y    = 0.75
        self.legend_X    = 0.70
        self.color_kpi   = ROOT.kBlue
        self.color_pipi  = ROOT.kRed
        self.color_kpisw = ROOT.kGreen
        self.color_ratio = ROOT.kBlack

        self.loadTrees()
        self.getVariableNames()

    def loadTrees(self):
        sweight = "(abs(B_M01-895.55)<100)*NB0_Kpigamma_sw"
        # imports data from the .root files that contain Monte Carlo data
        # imports data from the Kaon decays and adds them to a TTree object
        self.Tree_kpi  = ROOT.TChain("DecayTree")
        self.Tree_kpi.Add("../HistogramsLHCbData/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        self.Tree_kpi.Add("../HistogramsLHCbData/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        
        # imports data from the Pion decays and adds them to a TTree object
        self.Tree_pipi  = ROOT.TChain("DecayTree")
        self.Tree_pipi.Add("../HistogramsLHCbData/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        self.Tree_pipi.Add("../HistogramsLHCbData/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        
        # imports data from the Sample decays and adds them to a TTree object
        self.Tree_kpisw = ROOT.TChain("DecayTree")
        self.Tree_kpisw.Add("../HistogramsLHCbData/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root")

        #Defines the lengths of the TTree objects
        self.kpisw_size     = self.Tree_kpisw.GetEntries()
        self.Tree_kpi_size  = self.Tree_kpi.GetEntries()
        self.Tree_pipi_size = self.Tree_pipi.GetEntries()

    def getVariableNames(self):
        #Imports the variable names and finds which variables belong to each decay
        #Import all the Kaon decay variables
        variables_kpi = list(self.Tree_kpi.GetListOfBranches()) 
        variablesNames_kpi = [ str(var.GetFullName()) for var in variables_kpi]

        #Imports all the Pion decay variables
        variables_pipi = list(self.Tree_pipi.GetListOfBranches()) 
        variablesNames_pipi = [ str(var.GetFullName()) for var in variables_pipi]
        
        #uses sets to find the intersection and exclusive variables
        self.CommonVariables = list( set(variablesNames_pipi) & set(variablesNames_kpi) )
        self.kpiVariables    = list( set(variablesNames_kpi) - set(variablesNames_pipi) )
        self.pipiVariables   = list( set(variablesNames_pipi) - set(variablesNames_kpi) )
        self.AllVariables    = self.CommonVariables + self.kpiVariables + self.pipiVariables 

        self.CommonVariables.sort()
        self.kpiVariables   .sort()
        self.pipiVariables  .sort()
        self.AllVariables   .sort()

    def findVariableBound(self, variable):
        #Finds the maximium and minimum values for each variable
        #Creates a list of possible bounds
        HistBounds = [self.Tree_pipi.GetMinimum(variable), 
                      self.Tree_pipi.GetMaximum(variable),
                      self.Tree_kpi.GetMinimum(variable),
                      self.Tree_kpi.GetMaximum(variable)]
        HistBounds.sort()

        #Save the top and bottom entries of the sorted list
        self.lowerBound = float('%.1g' % HistBounds[0])
        self.upperBound = float('%.1g' % HistBounds[-1])

    def createKpiswHist(self, variable):
        self.findVariableBound(variable)
        h_kpisw = ROOT.TH1F(f"h_kpisw", f"{variable}",  self.NoOfBins, self.lowerBound, self.upperBound)
        h_kpisw.SetLineColor(self.color_kpisw)
        h_kpisw.SetLineWidth(2)
        self.Tree_kpisw.Draw(f"{variable}>>h_kpisw")

        self.h_kpisw = h_kpisw

    def createKpiHist(self, variable):
        self.findVariableBound(variable)
        h_kpi = ROOT.TH1F(f"h_kpi", f"{variable}",  self.NoOfBins, self.lowerBound, self.upperBound)
        h_kpi.SetLineColor(self.color_kpi)
        h_kpi.SetLineWidth(2)
        self.Tree_kpi.Draw(f"{variable}>>h_kpi")

        self.h_kpi = h_kpi

    def createPipiHist(self, variable):
        self.findVariableBound(variable)
        h_pipi = ROOT.TH1F(f"h_pipi", f"{variable}", self.NoOfBins, self.lowerBound, self.upperBound)
        h_pipi.SetLineColor(self.color_pipi)
        h_pipi.SetLineWidth(2)
        self.Tree_pipi.Draw(f"{variable}>>h_pipi")
        
        self.h_pipi = h_pipi

    def createRatioPlot(self):
        h_ratio = self.h_pipi.Clone("h_ratio")
        h_ratio.Divide(self.h_kpi)
        h_ratio.Scale(self.Tree_kpi_size/self.Tree_pipi_size)

        h_ratio.SetTitle("")
        h_ratio.SetLineColor(self.color_ratio)
        h_ratio.SetMarkerStyle(20)

        h_ratioy = h_ratio.GetYaxis()
        h_ratioy.SetTitle("Ratio")
        h_ratioy.SetTitleSize(20)
        h_ratioy.SetTitleFont(43)
        h_ratioy.SetLabelFont(43)
        h_ratioy.SetLabelSize(15)

        h_ratiox = h_ratio.GetXaxis()
        h_ratiox.SetLabelFont(43)
        h_ratiox.SetLabelSize(15)

        h_ratio.Draw("SAME")

        self.h_ratio = h_ratio
    
    def createLegend(self):
        legend = ROOT.TLegend(self.legend_X, self.legend_Y, self.legend_X + 0.15, self.legend_Y + 0.10)
        legend.AddEntry(self.h_pipi,"B^{0}#rightarrow #rho^{0}#gamma")
        legend.AddEntry(self.h_kpi, "B^{0}#rightarrow K^{*0}#gamma")
        legend.SetBorderSize(0)
        legend.SetTextSize(0.032)
    
        self.legend = legend

    def createSingleImage(self, variable, decay = "kpi"):
        self.canvas = ROOT.TCanvas("c", "canvas", self.CanvasSize, self.CanvasSize)
        
        if decay == "kpi" : 
            self.createKpiHist(variable)
            self.h_kpi.DrawNormalized("HISTO", norm=1)
        elif decay == "pipi" : 
            self.createPipiHist(variable)
            self.h_pipi.DrawNormalized("HISTO", norm=1)
        elif decay == "kpisw":
            self.createKpiswHist(variable)
            self.h_pipi.DrawNormalized("HISTO", norm=1)
        else:
            raise Exception(" \'decay\' must be ethier \'kpi(sw)\' or \'pipi\'.")

        #Draws and saves the full image
        self.canvas.SetGridx()
        self.canvas.Draw()
        self.canvas.Print(f"imgs/plot_{variable}_{decay}_single.pdf")
        self.canvas.Close()

    def createDoubleCanvas(self):
        canvas = ROOT.TCanvas("c", "canvas", self.CanvasSize, self.CanvasSize)

        padHist = ROOT.TPad("padHist", "padHist", 0, 0.3, 1, 1.0)
        padHist.SetBottomMargin(0.05)
        padHist.SetGridx()
        padHist.Draw()

        canvas.cd()
        padRatio = ROOT.TPad("padRatio", "padRatio", 0, 0.01, 1, 0.3)
        padRatio.SetTopMargin(0.05)
        padRatio.SetBottomMargin(0.2)
        padRatio.SetGridx()
        padRatio.Draw()

        self.canvas   = canvas
        self.padHist  = padHist
        self.padRatio = padRatio

    def createDoubleImage(self, variable):
        #Defines canvas
        ROOT.gStyle.SetOptStat(0)
        self.createDoubleCanvas()

        #Defines and Draws both histograms
        self.padHist.cd(0)
        self.createKpiHist(variable)
        self.createPipiHist(variable)

        #Sets the title and font size
        h_kpiy = self.h_kpi.GetYaxis()
        h_kpiy.SetTitle(f"{variable} distribution")
        h_kpiy.SetTitleSize(20)
        h_kpiy.SetTitleFont(43)
        h_kpiy.SetLabelFont(43)
        h_kpiy.SetLabelSize(15)

        h_kpix = self.h_kpi.GetXaxis()
        h_kpix.SetLabelFont(43)
        h_kpix.SetLabelSize(0.0)

        #Draws histograms
        self.h_kpi.DrawNormalized("HISTO", norm=1)
        self.h_pipi.DrawNormalized("HISTO SAME", norm=1)

        #Defines and Draws the legend
        self.createLegend()
        self.legend.Draw("SAME")

        #Defines and draws the ratio plot
        self.padRatio.cd(0)
        self.createRatioPlot()
        self.h_ratio.Draw("SAME")

        #Draws and saves the full image
        self.canvas.Draw()
        self.canvas.Print(f"imgs/plot_{variable}.pdf")
        self.canvas.Close()

    def createAllDoubleImages(self):
        ROOT.gROOT.SetBatch(True)
        for variable in self.CommonVariables:
            self.createDoubleImage(variable)

    def computeDifference(self, variable):
        #Returns the P value
        self.createPipiHist(variable)
        self.createKpiHist(variable)
        return self.h_kpi.Chi2Test(self.h_pipi)
    
    def computeAllDifferences(self):
        diffDict = {}
        for var in self.CommonVariables:
            diff = self.computeDifference(var)
            diffDict[var] = diff
        
        file = open("data/Chi2Values.csv", "wb")
        write = csv.DictWriter(file, diffDict.keys())
        write.writerows(diffDict)
        file.close()

    

if __name__ == "__main__":
    p = PlotCreator()
    print(p.kpisw_size)
