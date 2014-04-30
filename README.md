cs578-project3
=====
This is an implementation of the Naive Bayes and K-Nearest Neighbor machine learning classification algorithms used to derive a classification model for predicting a set of labels for classifying scrubbed medical patient records into one of three categories: Smokers, Non-Smokers or Unknown, by training a classifier model using an existing set of medical records.

Medical Record Format Schema:
<ROOT>
  <RECORD ID="1">
    <SMOKING STATUS="SMOKER"></SMOKING>
    <TEXT>Patient annotations</TEXT>
  </RECORD>
</ROOT>

Medical Record DTD:
<!DOCTYPE ScrubbedMedicalRecordSet [
<!ELEMENT ROOT (#PCDATA)>
<!ELEMENT RECORD (ID,SMOKING,TEXT)>
<!ATTLIST RECORD ID ID #REQUIRED>
<!ELEMENT SMOKING (SMOKING|NON-SMOKING|UNKNOWN)>
<!ATTLIST SMOKING STATUS CDATA #IMPLIED>
<!ELEMENT TEXT (#PCDATA)>
]>


##########
##########
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
usage: K-Nearest Neighbor Classifier [-h] [-t TRAININGSET] [-r TERMRANKINGS]
                                     [-a ASSOCFUNC] [-K KNEIGHBORS]
                                     [-s SIMILARITY_FUNC] [-S SAMPLETYPE]

Trains a K-Nearest Neighbor classifier to label patients as SMOKING, NON-
SMOKING or UNKNOWN based on available (scrubbed) medical record information.

optional arguments:
  -h, --help            show this help message and exit
  -t TRAININGSET, --trainingSet TRAININGSET
                        Path to the labeled medical record training set file,
                        e.g. ./path/to/training.txt
  -r TERMRANKINGS, --termrankings TERMRANKINGS
                        Path to term rankings pickle file, e.g.
                        ./path/to/termRankings.p
  -a ASSOCFUNC, --associationFunction ASSOCFUNC
                        The association function used to compare the relevancy
                        of a term to a specific class label [default=chi-
                        square|dice].
  -K KNEIGHBORS, --kNeighbors KNEIGHBORS
                        Total neighbors to sample.
  -s SIMILARITY_FUNC, --similarity SIMILARITY_FUNC
                        The similarity function used to compare a unlabeled
                        examples to a labeled kth-neighbor
                        [default=euclidean|manhatten|minkowski].
  -S SAMPLETYPE, --sampleType SAMPLETYPE
                        The method to sample 'K' records from each label
                        subset, top 'K'' are records with the max combined
                        term relevance score [default=Krandom|topK],' this
                        sample type doesn't apply when using the hamming
                        distance similarity.

