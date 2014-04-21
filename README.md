cs578-project3
=====
usage: Naive Bayes Classifier [-t TRAININGSET] [-m MU] [-s]

Trains a Naive Bayes classifier to label patients as SMOKING, NON-SMOKING or
UNKNOWN based on available (scrubbed) medical record information.

optional arguments:
  -t TRAININGSET, --trainingSet TRAININGSET
                        The path to the labeled medical record training set
                        file
  -m MU, --mu MU        Tuning parameter for the Naive Bayes classifier,
                        default is the length of the unique terms across all documents in the training set
  -s, --BayesianSmoothing
                        Specifies the Bayesian esimate for parameter smoothing
                        in the Naive Bayes classifier, default is Dirichlet
                        smoothing which considers mu
                        
##########
##########
usage: K-Nearest Neighbor Classifier [-t TRAININGSET] [-K KNEIGHBORS]

Trains a K-Nearest Neighbor classifier to label patients as SMOKING, NON-
SMOKING or UNKNOWN based on available (scrubbed) medical record information.

optional arguments:
  -t TRAININGSET, --trainingSet TRAININGSET
                        The path to the labeled medical record training set
                        file
  -K KNEIGHBORS, --kNeighbors KNEIGHBORS
                        Total neighbors to compare to in each class.

