#-------------------------------------------------------------------------------
# Name:        cs-578-project 3
# Purpose:     Trains a k-Nearest Neighbor classifier to apply labels to scrubbed 
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
import random

def hamdist(str1, str2):
    '''Count the # of differences between equal length 
    strings str1 and str2, borrowed from http://code.activestate.com/recipes/499304-hamming-distance/'''
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs;

def random_subset( iterator, K ):
    '''Implementation of resevoir sampling in Python borrowed from
        http://propersubset.com/2010/04/choosing-random-elements.html'''
    result = []
    N = 0
    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result

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

def computeSimilarityToKSamples( w,c,termClassOccurrenceLookup,totalTermsInTrainingDocPerClass,recordTermFeatureVector):
    ''' Computes the probability that term 'w' belongs to classl 'c' '''
    tf         = termClassOccurrenceLookup[w][c];
    cfw        = sum( i[1] for i in termClassOccurrenceLookup[w].iteritems() );
    # |c|
    _c         = math.fabs(totalTermsInTrainingDocPerClass[c])
    C          = sum(i[1] for i in totalTermsInTrainingDocPerClass.iteritems());
    P_wc = 0.0
    # V
    V          = float(len(recordTermFeatureVector))
    return P_wc;

def calcFeatureHammingDistance(a, b):
    '''Calculate total hamming distance for all terms in the feature vector
        TODO: Implement alternative similarity measure.  This initial approach is naive 
        in that we lose potentially relevant information from the longer term feature 
        vector when equalizing the vector lengths to compute their Hamming Distance similiarity.'''
    FeatureHammingDistance = 0.0;
    for x, y in zip(a, b):
        # terms as a list of binary encoded strings
        list1 = ' '.join(format(ord(xF), 'b') for xF in x).split(' ')
        list2 = ' '.join(format(ord(yF), 'b') for yF in y).split(' ')
        # match list lengths by setting lists equal to shortest list
        # NOTE:  Review since we are essentially trimming extra terms
        # from the record with the larrger term feature vector
        list1 = list1[0:len(list2)] if ( len(list2) < len(list1) ) else list1
        list2 = list2[0:len(list1)] if ( len(list1) < len(list2) ) else list2
        for subx, suby in zip(list1, list2):
            FeatureHammingDistance += hamdist(subx, suby)

    return FeatureHammingDistance;

if __name__ == '__main__':
    '''Train a Naive Bayes Classifier to classify 
        a set of (scrubbed) medical records with the label(s) 
        'SMOKER','NON-SMOKER' or 'UNKNOWN' '''
    parser = argparse.ArgumentParser(prog="K-Nearest Neighbor Classifier"
                                     ,description='Trains a K-Nearest Neighbor classifier '\
                                         'to label patients as SMOKING, NON-SMOKING or UNKNOWN based on available (scrubbed) medical record information.')
    parser.add_argument('-t', '--trainingSet'
                        ,help='The path to the labeled medical record training set file'
                        ,default=''
                        ,dest="trainingSet");
    parser.add_argument('-K','--kNeighbors'
                    ,dest="kNeighbors"
                    ,default = 3
                    ,type=int
                    ,help="Total neighbors to compare to in each class.");
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
        # For each document in the training set, compute similarity to 'K' nearest neighbors for each label
        K = args.kNeighbors
        # compute similarity vectors for each class label for each record
        # for each document in the training set...
        recordClassificationStats = {};
        for record in documentTuples:
            recordId                                 = record[0]
            recordLabel                              = record[1]
            recordTermFeatureVector                  = record[2].split(' ');
            recordClassificationStats[recordId]      = {'distance':{} , 'actual_label':recordLabel};
            # compute 'K' nearest neighbor
            # get k-random neighbors
            kNeighbors = random_subset( documentTuples, K )
            for kthNeighbor in kNeighbors:
                kthNeighborId                              = kthNeighbor[0];
                neighborTermFeatureVector                  = kthNeighbor[2].split(' ');
                # Use the Hamming Distance as a way to determine the similarity between this record's term
                # term feature vector nad it's kthNeighbor
                featureHammingDistance = 0.0;
                featureHammingDistance = calcFeatureHammingDistance(neighborTermFeatureVector, recordTermFeatureVector);
                recordClassificationStats[recordId]['distance'][kthNeighborId]     = featureHammingDistance;

        # get minimum distance from each neighbor
        finalLabels = {}
        for vector in recordClassificationStats.iteritems():
            # identify which neighbor was 'nearest' to
            # each example and output the nearest neighbor's actual class along with
            # the example's predicted class.
            distanceFromNeighbor      = vector[1]['distance']
            maxProb                   = min(distanceFromNeighbor.iteritems(), key=operator.itemgetter(1))[0]
            # TODO: refactor implementation
            predictedLabel = '';
            for doc in documentTuples:
                if doc[0] == maxProb:
                    predictedLabel = doc[1];
                    break;
            finalLabels[vector[0]]    = [predictedLabel,vector[1]['actual_label']];

        print "'{0}' records classified".format(len(finalLabels))
        actualLbls    = [];
        predictedLbls = [];
        for output in sorted(finalLabels.iteritems(), key=operator.itemgetter(0)):
            actualLbls.append( output[1][1] );
            if output[1][0] == output[1][1]:
                predictedLbls.append( output[1][0] );
            print "Record Id: '{0}' Classified as '{1}',Actual Classification: '{2}'".format(str(output[0]),output[1][0],output[1][1]);
        print "Correctly classified: '{0}' of '{1}' records".format( len(predictedLbls),len(actualLbls) );
