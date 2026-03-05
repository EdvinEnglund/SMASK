# SMASK

Repo for Projected in statistical machine learning 1RT700 at UU
Project group 30:

Edvin Englund
Emil Anbäcken Rudfeldt
Joakim Kilbo
Seleman Hassan

Description of files and scripts:

- 3_data_analysis.py loads and creates plots for the training data. Bar charts are perecentages of high/low demand per value of the feature
- Edvin_preprocessing.py runs the preprocessing and train/test splitting pipeline. This must be ran before any of the ML-training scripts can run.
- BenchmarkClassifier.py runs a random classifier based on class probabilities in training data.
- Emil.py is for the feature selection, tuning w. cross validation and final testing in the LDA algorithm implementation.
- Joakim.py is same as above for the ADABoost algorithm.
- files in knn folder are for training/tuning and testing of the knn implementation respectively.
- EdvinRForest.py is the feature selection, training, tuning w. cross validation for Random Forest Classifier.
- MainClassifier.py contains the full pipeline loading, preprocessing, training and final testing for the Random Classifier after it was chosen as the main model, based on the features and hyperparameters found in file above.
- Predictions.py is the full pipeline for loading, preprocessing, training and finally predicting high/low demand on the new data provided after the first submission based on the Random Forest Classifier trained on the entire original training set, in contrast to the 70% split used for model comparison.
