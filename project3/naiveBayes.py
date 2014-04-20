#-------------------------------------------------------------------------------
# Name:        cs-578-project 3
# Purpose:     Trains a Naive Bayes classifier to apply labels to scrubbed 
# medical records identfying patients as either 'smoking', 'non-smoking', 
# or 'unknown' class.
#   
# Author:      kingrichard2005
#
# Created:     20141604
# Copyright:   (c) kingrichard2005 2014
# Licence:     MIT
#-------------------------------------------------------------------------------
import re
import math
import operator
import os
import argparse

def getTotalNumTermsInTrainingDocPerClass(documentTuples,classificationLabels):
    '''Gets the total # of terms that occur in training documents with class label c'''
    documentsWithClassLabelC = {};
    returnReference = {}
    # Can be parallelized by class label
    for c in classificationLabels:
        documentsWithClassLabelC[c] = [dt[2] for dt in documentTuples if str(dt[1]) == str(c)];

    # accumulate all terms per class
    for label,terms in documentsWithClassLabelC.iteritems():
        returnReference[label] = len([i[0] for i in [term.split(' ') for term in terms] ]);

    return returnReference;

def getTermClassOccurenceLookup(documentTuples,uniqueTermsList):
    '''Gets the number of times the a term occurs in class 'c' of each document'''
    termClassOccurenceLookup = {};
    for t in documentTuples:
        # get content as list
        contentTermList = t[2].split(' ');
        # intersect to get unique terms in this document
        uqTermsInDocument = list(set(contentTermList) & set(uniqueTermsList));
        for uq in uqTermsInDocument:
            if termClassOccurenceLookup.has_key(uq) is False:
                termClassOccurenceLookup[uq]= {'SMOKER':0,'NON-SMOKER':0,'UNKNOWN':0};
            count = (t[1],1)
            termClassOccurenceLookup[uq][count[0]] += 1;
    return termClassOccurenceLookup;

def getUniqueTerms(documentTuples):
    uniqueTermList = []
    for x in documentTuples:
        [uniqueTermList.append( t ) for t in x[2].split(' ')];

    return list(set(uniqueTermList));

def removeNumbersAndPunctuation(documentTuples):
    """Filter numbers and punctuation"""
    localDocumentTuples = []
    for x in documentTuples:
        cleanString = " ".join(re.findall('[^0-9\.\-\s\/\:\,_\[\]\%]+',x[0][2]))
        localDocumentTuples.append( (x[0][0],x[0][1],cleanString) );

    return localDocumentTuples;

def getTrainingSetTuples(trainingSet):
    """"Each training example is represented as a 
        3-tuple with the following schema: 
        ( {Record ID}, {Training Label}, {Feature Term String - used for feature vector extraction} )."""
    recordIdTupleList = [];
    with open(trainingSet, 'r') as content_file:
        # Read records into list
        content    = content_file.read()
        content    = content.split("\n")
        content    = [m for m in content]
        tmpStr     = ''
        tmpStrList = []
        for m in content:
            if m == '<ROOT>':
                continue;
            elif m == '':
                tmpStrList.append(tmpStr);
                tmpStr = ''
            else:
                tmpStr += ' {0}'.format(m);

        # Extract tuples from parsed records
        pattern   = re.compile(R'<RECORD\sID="(\d*)">\s*<SMOKING\sSTATUS="(\S*)"></SMOKING>\s*<TEXT>\s*([^<>]+)\s*</TEXT>');
        recordIdTupleList = []
        for str in tmpStrList:
            recordIdTupleList.append([m for m in pattern.findall(str)]);
        recordIdTupleList = [t for t in recordIdTupleList if len(t) != 0];
    return recordIdTupleList;

def computeLabelPrediction( w,c,termClassOccurrenceLookup,totalTermsInTrainingDocPerClass,recordTermFeatureVector, mu = 0.75, useBayesianSmoothing = False ):
    ''' Computes the probability that term 'w' belongs to classl 'c' '''
    tf         = termClassOccurrenceLookup[w][c];
    cfw        = sum( i[1] for i in termClassOccurrenceLookup[w].iteritems() );
    # |c|
    _c         = math.fabs(totalTermsInTrainingDocPerClass[c])
    C          = sum(i[1] for i in totalTermsInTrainingDocPerClass.iteritems());
    P_wc = 0.0
    if useBayesianSmoothing:
        # V
        V          = float(len(recordTermFeatureVector))
        # Bayesian smoothing estimate
        P_wc       = ( tf + 1 ) / ( math.fabs(_c) + math.fabs(V) );
    else:
        # Dirichlet smoothing estimate
        P_wc      = ( tf + ( mu * ( cfw / C ) ) ) / ( _c + mu );
    return P_wc;

if __name__ == '__main__':
    '''Train a Naive Bayes Classifier to classify 
        a set of (scrubbed) medical records with the label(s) 
        'SMOKER','NON-SMOKER' or 'UNKNOWN' '''
    parser = argparse.ArgumentParser(prog="Naive Bayes Classifier"
                                     ,description='Trains a Naive Bayes classifier '\
                                         'to label patients as SMOKING, NON-SMOKING or UNKNOWN based on available (scrubbed) medical record information.')
    parser.add_argument('-t', '--trainingSet'
                        ,help='The path to the labeled medical record training set file'
                        ,default=''
                        ,dest="trainingSet");
    parser.add_argument('-m','--mu'
                        ,dest="mu"
                        ,type=float
                        ,help="Tuning parameter for the Naive Bayes classifier, default is the length of the unique terms across all documents in the training set");
    parser.add_argument('-s','--BayesianSmoothing'
                    ,dest="BayesianSmoothing"
                    ,action='store_true'
                    ,help="Specifies the Bayesian esimate for parameter smoothing in the Naive Bayes classifier, default is Dirichlet smoothing which considers mu");
    args = parser.parse_args();
    # check arguments, training set is required.
    if os.path.isfile( args.trainingSet ) is False:
        parser.print_help()
    else:
        ####
        # Collect required components
        documentTuples                        = getTrainingSetTuples(args.trainingSet);
        classificationLabels                  = ['SMOKER','NON-SMOKER','UNKNOWN']                                          # classificiation labels
        documentTuples                        = removeNumbersAndPunctuation(documentTuples);                               # Process record tuples
        uniqueTermsList                       = getUniqueTerms(documentTuples);                                            # get unique terms in problem space
        termClassOccurrenceLookup             = getTermClassOccurenceLookup(documentTuples,uniqueTermsList);               # get Tf_wc
        # get C = |c| = total number of terms that occur in training documents with class label 'c'
        totalTermsInTrainingDocPerClass       =  getTotalNumTermsInTrainingDocPerClass(documentTuples,classificationLabels)# Get total 'N'
        totalNtrainingInstances               = len(documentTuples);
        counts                                = { 'SMOKER'    : len([x for x in documentTuples if x[1] == 'SMOKER'])
                                                , 'NON-SMOKER': len([x for x in documentTuples if x[1] == 'NON-SMOKER' ])
                                                , 'UNKNOWN'   : len([x for x in documentTuples if x[1] == 'UNKNOWN']) 
                                                }
        # proportion 'P(c)' lookup of training instances with label 'c'
        proportionLookup = {
                             'SMOKER'    : float(counts['SMOKER'])     / float(totalNtrainingInstances)
                            ,'NON-SMOKER': float(counts['NON-SMOKER']) / float(totalNtrainingInstances)
                            ,'UNKNOWN'   : float(counts['UNKNOWN'])    / float(totalNtrainingInstances)
                            };
        ####
        # For each document in the training set, compute probability for each label
        # configurable tuning parameter...or do we set it to the total unique terms in the record collection??
        # TODO: confirm
        mu = args.mu if args.mu is not None else len(uniqueTermsList);
        # compute probability vectors P(C|D) for each class label for each record
        # for each document in the training set...
        recordClassificationStats = {};
        for record in documentTuples:
            recordId                                 = record[0]
            recordLabel                              = record[1]
            recordTermFeatureVector                  = record[2].split(' ');
            recordClassificationStats[recordId]      = {'predictions':{} , 'actual_label':recordLabel};
            # compute label probability vector for each document record ...
            for classLabel in classificationLabels:
                # P(c)
                classProportion       = proportionLookup[classLabel];
                probabilityListVector = []
                 # for each term in record feature vector
                 # compute P_wc label probability vector for all terms in document record event / occurrence space
                for term in recordTermFeatureVector:
                    # compute probability that term is related to class
                    P_wc = computeLabelPrediction( term
                                                  ,classLabel
                                                  ,termClassOccurrenceLookup
                                                  ,totalTermsInTrainingDocPerClass
                                                  ,recordTermFeatureVector
                                                  ,mu
                                                  ,useBayesianSmoothing = args.BayesianSmoothing 
                                                  );
                    probabilityListVector.append(P_wc);

                # aggregate product is the first element plus the product
                # we add zero prediction values for a term to the aggregate instead of multiplying
                aggregateProduct  = probabilityListVector[0]; 
                tmp               = reduce( lambda x,y: x * y if x > 0 and y > 0 else x + y , probabilityListVector[1:]);
                aggregateProduct  = ( (aggregateProduct * tmp) if aggregateProduct > 0.0 else (aggregateProduct + tmp) ) * classProportion;
                acc               = 0;
                for i in xrange(0,len(classificationLabels)):
                    acc += aggregateProduct;

                # avoid computing likelihood if the accumulated likelihood for all classes is 0.0
                computedLikelihood                                                 = 0.0 if acc == 0.0 else float(aggregateProduct / acc);
                recordClassificationStats[recordId]['predictions'][classLabel]     = computedLikelihood;

        # get maximum probability from each class
        finalLabels = {}
        for vector in recordClassificationStats.iteritems():
            # get feature vectors
            predictedOutcomesPerClass = vector[1]['predictions']
            maxProb                   = max(predictedOutcomesPerClass.iteritems(), key=operator.itemgetter(1))[0]
            finalLabels[vector[0]]    = [maxProb,vector[1]['actual_label']];

        print "'{0}' records classified".format(len(finalLabels))
        actualLbls    = [];
        predictedLbls = [];
        for output in sorted(finalLabels.iteritems(), key=operator.itemgetter(0)):
            actualLbls.append( output[1][1] );
            if output[1][0] == output[1][1]:
                predictedLbls.append( output[1][0] );
            print "Record Id: '{0}' Classified as '{1}',Actual Classification: '{2}'".format(str(output[0]),output[1][0],output[1][1]);
        print "Correctly classified: '{0}' of '{1}' records".format( len(predictedLbls),len(actualLbls) );