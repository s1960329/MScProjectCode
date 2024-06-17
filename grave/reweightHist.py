import numpy as np
import pandas as pd
import uproot
from hep_ml import reweight
from hep_ml.metrics_utils import ks_2samp_weighted
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



columns = ["nTracks","B_P","B_Cone3_B_ptasy","B_MINIPCHI2","B_FD_OWNPV","gamma_PT","piminus_PT"]

with uproot.open("../data/kpiG_MC_Bd2KstGamma_HighPt_prefilter_2018_noPIDsel-magup.root") as original_file:
        original_tree = original_file["DecayTree"]
        original = original_tree.arrays(columns, library="pd")

with uproot.open("../data/Sample_Kpigamma_2018_selectedTree_with_sWeights_Analysis_2hg_Unbinned-Mask1.root") as target_file:
        target_tree = target_file["DecayTree"]
        target = target_tree.arrays(columns + ["NB0_Kpigamma_sw"], library="pd", cut = "(abs(B_M01-895.55)<100)")

original_weights = np.ones(len(original))
target_weight = target["NB0_Kpigamma_sw"]



original_train, original_test = train_test_split(original)
target_train, target_test = train_test_split(target)

original_weights_train = np.ones(len(original_train))
original_weights_test = np.ones(len(original_test))

hist_settings = {'bins': 100, 'density': True, 'alpha': 0.7}

def draw_distributions(original, target, new_original_weights):
    plt.figure(figsize=[15, 7])
    for id, column in enumerate(columns, 1):
        print(column)
        xlim = np.percentile(np.hstack([target[column]]), [0.01, 99.99])
        plt.subplot(2, 3, id)
        plt.hist(original[column], weights=new_original_weights, range=xlim, **hist_settings)
        plt.hist(target[column], range=xlim, **hist_settings)
        plt.title(column)
        plt.savefig(f"../imgs/plot_{columns}_reweight.pdf", dpi=227)
        print("KS over ", column, " = ", ks_2samp_weighted(original[column], target[column],
                                         weights1=new_original_weights, weights2=np.ones(len(target), dtype=float)))
        

draw_distributions(original, target, original_weights)

print(f"Original train shape: {original_train.shape}")
print(f"Target train shape: {target_train[columns].shape}")
print(f"Original weights train shape: {original_weights_train.shape}")
print(f"Original weights test shape: {original_weights_test.shape}")

bins_reweighter = reweight.BinsReweighter(n_bins=20, n_neighs=1.)
bins_reweighter.fit(original_train, target_train[columns], original_weight = original_weights_train, target_weight= target_train["NB0_Kpigamma_sw"]  )

bins_weights_test = bins_reweighter.predict_weights(original_test)
# validate reweighting rule on the test part comparing 1d projections
draw_distributions(original_test, target_test, bins_weights_test)