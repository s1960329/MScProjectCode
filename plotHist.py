from typing import Any
import ROOT
import csv

class PlotCreator():

    def __init__(self) -> None:
        ROOT.gROOT.SetBatch(True)

        self.CANVAS_SIZE_X   = 800
        self.CANVAS_SIZE_Y   = 800
        self.N_BINS          = 50
        self.LEGEND_X        = 0.70
        self.LEGEND_Y        = 0.75
        self.LEGEND_SIZE_X   = 0.1
        self.LEGEND_SIZE_Y   = 0.1
        self.HIST_LINE_WIDTH = 2
        self.FONT            = 43
        self.AXIS_TITLE_SIZE = 20
        self.LABEL_SIZE      = 15
        self.MARKER_STYLE    = 20

        self.COLOR           = {"kpi"  : ROOT.kBlue,
                                "pipi" : ROOT.kRed,
                                "kpisw": ROOT.kOrange,
                                "ratio": ROOT.kBlack}
        
        self.loadTrees()
        self.getVariableNames()

    def loadTrees(self):
        # Uses sweights to remove background from sample data
        sweight = "(abs(B_M01-895.55)<100)*NB0_Kpigamma_sw"
        self.Trees = {}

        # imports data from the .root files that contain Monte Carlo data
        # imports data from the Kaon decays and adds them to a TTree object
        self.Tree_kpi  = ROOT.TChain("DecayTree")
        self.Tree_kpi.Add("../HistogramsLHCbData/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        self.Tree_kpi.Add("../HistogramsLHCbData/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        self.Trees["kpi"] = self.Tree_kpi

        # imports data from the Pion decays and adds them to a TTree object
        self.Tree_pipi  = ROOT.TChain("DecayTree")
        self.Tree_pipi.Add("../HistogramsLHCbData/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        self.Tree_pipi.Add("../HistogramsLHCbData/pipiG_MC_Bd2RhoGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        self.Trees["pipi"] = self.Tree_pipi

        # imports data from the Sample decays and adds them to a TTree object
        self.Tree_kpisw = ROOT.TChain("DecayTree")
        self.Tree_kpisw.Add("../HistogramsLHCbData/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root")
        self.Trees["kpisw"] = self.Tree_kpisw

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

    def createHist(self, decay, variable):
        self.findVariableBound(variable)
        hist = ROOT.TH1F(f"h_{variable}_{decay}", f"{variable}_{decay}",  self.N_BINS, self.lowerBound, self.upperBound)
        hist.SetLineColor(self.COLOR[decay])
        hist.SetLineWidth(self.HIST_LINE_WIDTH)
        self.Trees[decay].Draw(f"{variable}>>h_{variable}_{decay}")
        self.Histograms[decay] = hist

    def createRatioPlot(self):
        #Defines and normalises Ratio plot
        h_ratio = self.Histograms["pipi"].Clone("h_ratio")
        h_ratio.Divide(self.Histograms["kpi"])
        h_ratio.Scale(self.Tree_kpi_size/self.Tree_pipi_size)

        # Style for Ratio plot
        h_ratio.SetTitle("")
        h_ratio.SetLineColor(self.COLOR["ratio"])
        h_ratio.SetMarkerStyle(self.MARKER_STYLE)

        # Style for y axis of Ratio plot
        h_ratioy = h_ratio.GetYaxis()
        h_ratioy.SetTitle("Ratio")
        h_ratioy.SetTitleSize(self.AXIS_TITLE_SIZE)
        h_ratioy.SetTitleFont(self.FONT)
        h_ratioy.SetLabelFont(self.FONT)
        h_ratioy.SetLabelSize(self.LABEL_SIZE)

        # Style for x axis of Ratio plot
        h_ratiox = h_ratio.GetXaxis()
        h_ratiox.SetLabelFont(self.FONT)
        h_ratiox.SetLabelSize(self.LABEL_SIZE)
        h_ratio.Draw("SAME")
        
        self.Histograms["ratio"] = h_ratio
    
    def createLegend(self):
        legend = ROOT.TLegend(self.LEGEND_X, self.LEGEND_Y, self.LEGEND_X + self.LEGEND_SIZE_X, self.LEGEND_Y + self.LEGEND_SIZE_Y)
        legend.AddEntry(self.Histograms["pipi" ], "B^{0}#rightarrow #rho^{0}#gamma - MC")
        legend.AddEntry(self.Histograms["kpi"  ], "B^{0}#rightarrow K^{*0}#gamma - MC")
        legend.AddEntry(self.Histograms["kpisw"], "B^{0}#rightarrow K^{*0}#gamma - Sample")
        legend.SetBorderSize(0)
        legend.SetTextSize(0.032)
    
        self.legend = legend

    def createSingleImage(self, variable, decay = "kpi"):
        self.canvas = ROOT.TCanvas("c", "canvas", self.CANVAS_SIZE_X, self.CANVAS_SIZE_Y)
        self.createHist(decay, variable)
        self.Histograms[decay].DrawNormalized("HISTO", norm=1)
        #Draws and saves the full image
        self.canvas.SetGridx()
        self.canvas.Draw()
        self.canvas.Print(f"../HistogramsLHCbImgs/plot_{variable}_{decay}_single.pdf")
        self.canvas.Close()

    def createDoubleCanvas(self):
        canvas = ROOT.TCanvas("c", "canvas", self.CANVAS_SIZE_X, self.CANVAS_SIZE_Y)

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
        self.createHist("kpi"  , variable)
        self.createHist("pipi" , variable)
        self.createHist("kpisw", variable)

        self.Histograms["kpi"].SetTitle(variable)

        #Sets the title and font size
        h_kpiy = self.Histograms["kpi"].GetYaxis()
        h_kpiy.SetTitle(f"{variable} distribution")
        h_kpiy.SetTitleSize(self.AXIS_TITLE_SIZE)
        h_kpiy.SetTitleFont(self.FONT)
        h_kpiy.SetLabelFont(self.FONT)
        h_kpiy.SetLabelSize(self.LABEL_SIZE)

        h_kpix = self.Histograms["kpi"].GetXaxis()
        h_kpix.SetLabelFont(self.FONT)
        h_kpix.SetLabelSize(0.0)

        #Draws histograms
        self.Histograms["kpi"  ].DrawNormalized("HISTO", norm=1)
        self.Histograms["kpisw"].DrawNormalized("HISTO SAME", norm=1)
        self.Histograms["pipi" ].DrawNormalized("HISTO SAME", norm=1)

        #Defines and Draws the legend
        self.createLegend()
        self.legend.Draw("SAME")

        #Defines and draws the ratio plot
        self.padRatio.cd(0)
        self.createRatioPlot()
        self.Histograms["ratio"].Draw("SAME")

        #Draws and saves the full image
        self.canvas.Draw()
        self.canvas.Print(f"../HistogramsLHCbImgs/plot_{variable}.pdf")
        self.canvas.Close()

    def createAllDoubleImages(self):
        for variable in self.CommonVariables:
            self.createDoubleImage(variable)

    def computeDifference(self, variable):
        #Returns the P value
        self.createHist("pipi",variable)
        self.createHist("kpi", variable)
        return self.Histograms["kpi"].Chi2Test(self.Histograms["pipi"])
    
    def computeAllDifferences(self):
        diffDict = {}
        for var in self.CommonVariables:
            diff = self.computeDifference(var)
            diffDict[var] = diff
        
        file = open("../HistogramsLHCbData/Chi2Values.csv", "wb")
        write = csv.DictWriter(file, diffDict.keys())
        write.writerows(diffDict)
        file.close()

    

if __name__ == "__main__":
    p = PlotCreator()
    p.createAllDoubleImages()
