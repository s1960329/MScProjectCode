from typing import Any
import ROOT
import csv
import math

class PlotCreator():

    def __init__(self) -> None:

        ROOT.gROOT.SetBatch(True)

        self.CANVAS_SIZE_X   = 800
        self.CANVAS_SIZE_Y   = 800
        self.N_BINS          = 250
        self.LEGEND_X        = 0.8
        self.LEGEND_Y        = 0.92
        self.LEGEND_SIZE_X   = 0.1
        self.LEGEND_SIZE_Y   = 0.07
        self.HIST_LINE_WIDTH = 2
        self.FONT            = 43
        self.AXIS_TITLE_SIZE = 20
        self.LABEL_SIZE      = 15
        self.MARKER_STYLE    = 50
        self.MARKER_SIZE     = 0.1
        self.UPPER_BOUND_Y   = 1
        self.LOWER_BOUND_Y   = 0
        self.UPPER_BOUND_X   = 1
        self.LOWER_BOUND_X   = 0
        
        self.COLOR           = {"kpi"  : ROOT.kBlue,
                                "kpisw": ROOT.kOrange,
                                "ratio": ROOT.kBlack}
        
        self.sweights   = "(abs(B_M01-895.55)<100)*NB0_Kpigamma_sw"
        self.trueid     = "(B_BKGCAT==0 ||B_BKGCAT== 50)"
        
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
        Tree_kpi.Add("../HistogramsLHCbData/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magdown.root")
        Tree_kpi.Add("../HistogramsLHCbData/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root")
        
        # imports data from the Sample decays and adds them to a TTree object
        Tree_kpisw = ROOT.TChain("DecayTree")
        Tree_kpisw.Add("../HistogramsLHCbData/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root")
        
        # stores trees
        self.Trees["kpi"  ] = Tree_kpi
        self.Trees["kpisw"] = Tree_kpisw

        # defines the lengths of the TTree objects
        self.Sizes["kpisw"] = self.Trees["kpisw"].GetEntries()
        self.Sizes["kpi  "] = self.Trees["kpi"  ].GetEntries()

    def getVariableNames(self):
        #Imports the variable names and finds which variables belong to each decay
        #Import all the Kaon decay variables
        variables_kpi = list(self.Trees["kpi"].GetListOfBranches()) 
        variablesNames_kpi = [ str(var.GetFullName()) for var in variables_kpi]

        #Imports all the Sample decay variables
        variables_kpisw = list(self.Trees["kpisw"].GetListOfBranches()) 
        variablesNames_kpisw = [ str(var.GetFullName()) for var in variables_kpisw]
        
        #uses sets to find the intersection and exclusive variables
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
        self.lowerBound = float('%.1g' % max(self.Trees["kpisw"].GetMinimum(variable), self.Trees["kpisw"].GetMinimum(variable), self.Trees["kpi"  ].GetMinimum(variable), self.Trees["kpi"  ].GetMinimum(variable)) )
        self.upperBound = float('%.1g' % min(self.Trees["kpi"  ].GetMaximum(variable), self.Trees["kpi"  ].GetMaximum(variable), self.Trees["kpisw"].GetMaximum(variable), self.Trees["kpisw"].GetMaximum(variable) ) )


    def createHist(self, variable, decay = "kpi"):
        #Computes variable bounds
        self.findVariableBounds(variable)
        
        #Defines variable histogram
        hist_name = f"h_{variable}_{decay}"
        hist = ROOT.TH1F(hist_name, f"{variable}_{decay}",  self.N_BINS, self.lowerBound, self.upperBound)
        
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
        self.canvas = ROOT.TCanvas(f"c_{variable}_{decay}", f"canvas_{variable}_{decay}", self.CANVAS_SIZE_X, self.CANVAS_SIZE_Y)
        
        #Defines the histogram
        self.createHist(decay, variable)
        self.Histograms[decay].DrawNormalized("HISTO", norm=1)
        
        #Draws and saves the full image
        self.canvas.SetGridx()
        self.canvas.Draw()
        self.canvas.Print(f"../HistogramsLHCbImgs/plot_{variable}_{decay}_single.pdf")
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
        h_ratio.SetLineColor(self.COLOR["ratio"])
        h_ratio.SetMarkerStyle(self.MARKER_STYLE)
        h_ratio.SetMarkerSize(self.MARKER_SIZE)

        self.UPPER_BOUND_ratio_Y = h_ratio.GetMaximum()
        self.LOWER_BOUND_ratio_Y = h_ratio.GetMinimum()
        h_ratio.SetAxisRange(self.LOWER_BOUND_ratio_Y, self.UPPER_BOUND_ratio_Y, "Y")

        # Style for y axis of Ratio plots
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
        self.UPPER_BOUND_X = h_ratiox.GetBinCenter(h_ratio.FindLastBinAbove())
        self.LOWER_BOUND_X = h_ratiox.GetBinCenter(h_ratio.FindFirstBinAbove())

        print("X:", self.LOWER_BOUND_X, self.UPPER_BOUND_X)
        
        h_ratio.Draw("SAME")
        h_ratiox.SetRangeUser(self.LOWER_BOUND_X, self.UPPER_BOUND_X)
        
        #Saves histogram 
        self.Histograms["ratio"] = h_ratio
    
    def createLegend(self):
        legend = ROOT.TLegend(self.LEGEND_X, self.LEGEND_Y, self.LEGEND_X + self.LEGEND_SIZE_X, self.LEGEND_Y + self.LEGEND_SIZE_Y)
        legend.AddEntry(self.Histograms["kpi"  ], "B^{0}#rightarrow K^{*0}#gamma - MC")
        legend.AddEntry(self.Histograms["kpisw"], "B^{0}#rightarrow K^{*0}#gamma - Sample")
        legend.SetBorderSize(0)
        legend.SetTextSize(0.032)
    
        self.legend = legend

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
        self.createHist(variable, "kpi")
        self.createHist(variable, "kpisw")

        self.Histograms["kpi"  ].SetTitle(variable)

        #Sets the title and font size
        h_kpiy = self.Histograms["kpi"].GetYaxis()
        h_kpiy.SetTitle(f"{variable} distribution")
        h_kpiy.SetTitleSize(self.AXIS_TITLE_SIZE)
        h_kpiy.SetTitleFont(self.FONT)
        h_kpiy.SetLabelFont(self.FONT)
        h_kpiy.SetLabelSize(self.LABEL_SIZE)
        
        h_kpix = self.Histograms["kpi"].GetXaxis()
        h_kpix.SetLabelSize(0.0)
        
        integral_kpi   = self.Histograms["kpi"  ].Integral()
        integral_kpisw = self.Histograms["kpisw"].Integral()

        try:
            maxBinContent_kpi   = self.Histograms["kpi"  ].GetBinContent(self.Histograms["kpi"  ].GetMaximumBin()) / integral_kpi
            maxBinContent_kpisw = self.Histograms["kpisw"].GetBinContent(self.Histograms["kpisw"].GetMaximumBin()) / integral_kpisw
            
            self.Histograms["kpi"  ].DrawNormalized("HISTO", norm=1)
            self.Histograms["kpisw"].DrawNormalized("HISTO SAME", norm=1)
            
            print(variable)
            print("Y:", 0, max(maxBinContent_kpi,maxBinContent_kpisw))
            
            #self.Histograms["kpisw"].SetAxisRange(self.LOWER_BOUND_X, self.UPPER_BOUND_X, "X")

            #Defines and Draws the legend
            self.createLegend()
            self.legend.Draw("SAME")

            #Defines and draws the ratio plot
            self.padRatio.cd(0)
            self.createRatioPlot(variable)
            self.Histograms["ratio"].Draw("SAME")

            #Draws and saves the full image
            self.canvas.Draw()

            self.canvas.Print(f"../HistogramsLHCbImgs/plot_{variable}.pdf")
            self.canvas.Close()
        except:
            pass
        

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
        
        file = open("../HistogramsLHCbData/Chi2Values.csv", "wb")
        write = csv.DictWriter(file, diffDict.keys())
        write.writerows(diffDict)
        file.close()


if __name__ == "__main__":
    p = PlotCreator()
    p.createDoubleImage("B_BMassFit_Kst_892_0_Kplus_PX")
    p.createDoubleImage("B_BMassFit_Kst_892_0_piminus_PE")
    # print(p.LOWER_BOUND_X, p.UPPER_BOUND_X)
    # p.createAllDoubleImages()
    # B_BMassFit_Kst_892_0_Kplus_PX
    # B_Cone2_B_pt