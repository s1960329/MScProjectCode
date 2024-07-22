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

24/06/2024
TODO: 
Improve GBDT model and continue to tune variables x
Deduce why reweighting does not work with some variables ~
Take the logarithms of some variables x
Fix ratio plots and limits x
Implement units (MeV/c)
Create one big image with a pdf file x
two or three slides summariing what I've done
Implement ROC curve 
Plot with new variables that Lais sending
Update with Lais on the variables that are not good 
Adapt code for pipi gamma, Look out for Lais's code

25/06/2024
Find a better way of scoring the reweighting x
Why does gamma_PT
Update with Lais on the variables that are not good 
Adapt code for pipi gamma, Look out for Lais's code

change from KS to chi2 x
Implement ROC curve 
adapt for pipi gamma
Neural Networks root, tensorflow, skilearn

When taking the logs of variables how should we handle when the value equals zero


history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=2,batch_size=600, epochs=100)


Questions

How to deal with numbers that are zero when taking the natural logarithm 
How accurate do the Boosest Decision Trees need to be in order to allow for classification
Is the a way to save a load machine learning models


02/07/2024
This week invovled creating and expereimenting with classifiers, most notably a neural network classifier, but also expereimtning with ADABoost, Stochastic Gradient Decent, and Random Forrests. With the Neural Network I first normalised the data so that the highest value would equal one and the lowest value would equal zero. I then trained the network to distingish between Monte Carlo data and Sample data which it was able to do after training to an accuracy of 70%. However, when thesample model was create but trained on the Gradient boosted decision trees calculated weights it was not possible to train the network much futher than an accuray of 50% Thus confirming that as far as the classifier is concerned the reweighted Monte carlo data is close to indistingaisble to the real sample data. This can be further seen in the ROC curve where the model with weights gives a far less accraute result. The next steps will be to classify the B Decays and apply machine learning models that allow for tagging.

04/07/2024
These next few days have been incredibly frustrating, The  end of the week has the target of getting a rough classifier in place that can seperate signal from background to do this I am having to adapt some c++ code that lais gave me. However, we're running into several problems with the code actually running, what's more, it requires the root files to have the GBReweighter variables, and for some reason they're not saving correctly with fewer entries than the actual files. Thankfully lais has been super helpful and is abe to trouble shoot my code for me, and the issue with fewer datapoints can be fixed later. What's important now is creating the classifier before friday afternoon.

08/07/2024
 - Should the testing and training data be seperated in the reweighter? (maybe test it but not crutial)
 - How should I better evaluate the classifiers?
 - Would it be more accurate if I had more data?
 - Hidden Layers for neural network?
 - How many samples per an epoch?
 - Random Forrest is the best?
 - should I have a 50/50 split of isMonteCarlo Data?  

13/07/2024
This week I tested various parametres for the classification models and wrote up a large amount of the project into the final .py classification files. I first experimented with overfitting tests and found the largest possible values that gave the most accurate result while also preventing overfitting. Once I had done this I testesd various variables and found the sets that gave the highest accuracy. What's more I then created a system that can be used to find a probabilistic measure of the neural network and find which probabilitic measure gives the highest accuracy. I'm also learning the basics of manim with the hope that it will help me with my oral presentation. My next steps will be to compare my results to the basic ROOT results and aim to get a greater accuracy.

16/07/2024
- Which different models could we use that are different enoguh for the votes
- Why does the threshold result give such wildly different answers
- Should I save all the models with joblib
- How much more accurate would the model be with more data?
- 



# with uproot.open(fileNamesRoot["f2"]) as TChain: # type: ignore
#     TTree = TChain["dataset"]
#     TestTree = TTree["TestTree"]
#     TrainTree = TTree["TrainTree"]


# testVariables  = TestTree.keys()
# trainVariables = TrainTree.keys()


# testRootFile = TestTree.arrays(testVariables, library="pd", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
# testRootFile = testRootFile.reset_index(drop=True)

# trainRootFile = TrainTree.arrays(trainVariables, library="pd", aliases ={"B_ETA": "-log(tan(atan(B_PT/B_PZ)/2))"}) # type: ignore
# trainRootFile = trainRootFile.reset_index(drop=True)



# print(len( trainRootFile[ trainRootFile["className"] == "Signal" ] ))
# print(len( trainRootFile[ trainRootFile["className"] == "Background" ] ))
# print()
# print(trainRootFile)
# print()
# print(testRootFile)


# testRootFile.to_csv(fileNamesCSV["f2v"])
# trainRootFile.to_csv(fileNamesCSV["f2t"])