# LHCb_BD_tagging
A series of machine learning models that can tag the radiative b decay in LHCb data


- GB Tuning Results 2
KSR, Runtime
(n_estimators=50,  learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.0546818322682516, 21
(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.182082594028184,  58
(n_estimators=150, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1896635168924834, 67
(n_estimators=250, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1979912075751054, 106
(n_estimators=500, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1934403664209463, 250
(n_estimators=250, learning_rate=0.1, max_depth=5, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1856314302611457, 148
(n_estimators=250, learning_rate=0.1, max_depth=2, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1931894301576618, 80
(n_estimators=250, learning_rate=0.1, max_depth=4, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.2026393734509542, 135
(n_estimators=250, learning_rate=0.2, max_depth=4, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1740906205481247, 121
(n_estimators=250, learning_rate=0.08,max_depth=4, min_samples_leaf=1000, gb_args={'subsample': 0.4}), 1.1962367009262824 ,139

- GB Tuning with folds 2
KSR, n_folds, Runtime
(n_estimators=250, learning_rate=0.1, max_depth=4, min_samples_leaf=1000, gb_args={'subsample': 0.4}), , 4,  


- GB Tuning Results 1
1  (n_estimators=50,  learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.8377414300305108, 23s
2  (n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.8983630641238445, 44s
3  (n_estimators=150, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.9193037762647586, 69s
4  (n_estimators=50,  learning_rate=0.5, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.8709630911609474, 26s
5  (n_estimators=50,  learning_rate=0.01,max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.2866531194138445, 27s
6  (n_estimators=500, learning_rate=0.01,max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.8276735187432498, 240s
7  (n_estimators=500, learning_rate=0.01,max_depth=5, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.8948980767880080, 357s
8  (n_estimators=150, learning_rate=0.1, max_depth=5, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.9071763533076986, 109s
9  (n_estimators=150, learning_rate=0.1, max_depth=5, min_samples_leaf=1000, gb_args={'subsample': 0.6}), KSR 0.8425944965196068, 174s
10 (n_estimators=150, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.6}), KSR 0.8937413269688022, 92s
11 (n_estimators=150, learning_rate=0.1, max_depth=5, min_samples_leaf=1000, gb_args={'subsample': 0.2}), KSR 0.8425944965196068, 45s
12 (n_estimators=150, learning_rate=0.1, max_depth=3, min_samples_leaf=10000,gb_args={'subsample': 0.4}), KSR 0.9144680049072742, 70s
13 (n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.9118033141434384, 95s
14 (n_estimators=125, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.8985383911237783, 60s
15 (n_estimators=175, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), KSR 0.9079390675493116, 83s

- GB Tuning with folds 1
(n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 0.4}), n_folds=4, KSR 0.9194408356956647, 273s

- GB tuning without gamma_PT
With Gamma    : 1.044123600339418
Without Gamma : 1.0703022654661998
for all variables we care about

Conclusions: Performance is slightly better without gamma_PT being trained on
             Performance is increased until the following points, n_estimators : ,




- Development logs

28/05/2024
Development began on the 24/05/2024 with initial sample code. This code plotted several histograms from provided .root files and was used as a starting point.
After familarising myself with PyROOT I began to work with the sample code to create more histograms and experiment with PyRoot syntax. So far I have created histograms of
all suggested variables which has aided in seperating these reactions. Next, I plan to inspect variables with TBrowser and create more histograms, ensuring that I overlay some of them, and create a general system that create histgrams of any two variables

20/06/2024
Development has been going smoothly, after drawing histograms with the inital sample code the uproot module was installed which allowed for use of matplotlib and pandas. Once histograms with these where created a Gradient boosted reweighter was implemented which allowed the Monte carlo data to look similar to the Sample Data. The Gradient Boosted De sion trees will not be tuned as to find the optimal value for the 

