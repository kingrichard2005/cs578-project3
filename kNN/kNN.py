#-------------------------------------------------------------------------------
# Name:        cs-578-project 3
# Description:     A prototype trainer for a K-Nearest Neighbor classifier that 
# attempts to predict smoaking status for unlabeled patients as one of three categories: 'SMOKER, 'NON-SMOKER'
# , or 'UNKNOWN'.  Classifier is trainied using a precompiled collection of labeled training record data
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
import cPickle as pickle

def random_subset( sampleList, K ):
    '''(REVIEW) Implementation of resevoir sampling in Python borrowed from
        http://propersubset.com/2010/04/choosing-random-elements.html'''
    try:
        result = []
        N = 0
        for item in sampleList:
            N += 1
            if len( result ) < K:
                result.append( item )
            else:
                s = int(random.random() * N)
                if s < K:
                    result[ s ] = item

        return result
    except:
        print "error sampling random subset from record tuple collection"

def getTotalNumTermsInTrainingDocPerClass(documentTuples,classificationLabels):
    '''Gets the total # of terms that occur in training documents with class label c'''
    try:
        documentsWithClassLabelC = {};
        returnReference = {}
        # Can be parallelized by class label
        for c in classificationLabels:
            documentsWithClassLabelC[c] = [dt[2] for dt in documentTuples if str(dt[1]) == str(c)];

        # accumulate all terms per class
        for label,terms in documentsWithClassLabelC.iteritems():
            returnReference[label] = len([i[0] for i in [term.split(' ') for term in terms] ]);

        return returnReference;
    except:
        print "error getting TotalNumTermsInTrainingDocPerClass"

def getUniqueTerms(documentTuples):
    uniqueTermList = []
    for x in documentTuples:
        [uniqueTermList.append( t ) for t in x[2].split(' ')];

    return list(set(uniqueTermList));

def removeNumbersAndPunctuation(documentTuples):
    """Filter numbers and punctuation"""
    try:
        localDocumentTuples = []
        for x in documentTuples:
            cleanString = " ".join(re.findall('[^0-9\.\-\s\/\:\,_\[\]\%]+',x[0][2]))
            localDocumentTuples.append( (x[0][0],x[0][1],cleanString) );

        return localDocumentTuples;
    except:
        print "error removing numbers and punctuation"

def getTrainingSetTuples(trainingSet):
    """"Each training example is represented as a 
        3-tuple with the following schema: 
        ( {Record ID}, {Training Label}, {Feature Term String - used for feature vector extraction} )."""
    try:
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
    except:
        print "error getting record tuples from training set file {0}".format(str(trainingSet));

def calculateVectorSimilarity(x, y):
    '''Calculate the similarity between two vectors using the Euclidean Distance function.'''
    try:
        euclideanDistance = 0.0;
        localX = x[:];
        localY = y[:];
        sumAcc = 0;
        # if lengths differ
        # pad shorter vector with zeroes
        if abs(len(x)-len(y)) != 0:
            zeroPad = [0] * abs(len(x)-len(y));
            if ( len(x) < len(y) ):
                sumAcc += len(y)
                for e in zeroPad:
                    localX.append(e);
            else:
                sumAcc += len(x)
                for e in zeroPad:
                    localY.append(e);
        for a in xrange(0,sumAcc):
            euclideanDistance += float( (localX[a]-localY[a])**2 );
        euclideanDistance = math.sqrt(euclideanDistance);
        # return distance measure rounded to 4 decimal places
        return round(euclideanDistance,4);
    except:
        print "error calculating similarity score"

def calcChiSquare(n_a,n_b,n_ab,N):
    '''Calculates the Chi-Square and returns the float value rounded to 4 decimal places
        see Search Engines: Information Retrieval in Practice ( Ch 6.2 )'''
    try:
        result = float( (n_ab - (float(1/N) * n_a * n_b)  )**2 )  / (n_a * n_b);
        return round(result,4)
    except:
        print "error calculating chiSquare"

def calcDiceCoeff(n_a,n_b,n_ab):
    '''Calculates the Dice coefficient and returns the float value rounded to 4 decimal places
        see Search Engines: Information Retrieval in Practice ( Ch 6.2 )'''
    try:
        #result = float( n_ab  / n_a + n_b );silly me
        result = float( n_ab )  / float( n_a + n_b )
        return round(result,4)
    except:
        print "error calculating DiceCoeff"

def getTermRanksPerClass(uniqueTermsList , classificationLabels, documentTuples):
    '''Construct a lookup of term rankings per class / label'''
    try:
        # schema
        # { {SMOKING_STATUS}:[Term rankings by descending order] }
        termRankingsPerClass = {};

        # for each class
        for classLabel in classificationLabels:
            classLbelRecordSubset = [t for t in documentTuples if t[1] == classLabel];
            # n_b: num. of records containing class label (n_b)
            n_b  = len(classLbelRecordSubset)
            # rank each unique term
            for term in uniqueTermsList:
                termRecordSubset = [t for t in documentTuples  if t[2].find(term) != -1];
                # n_a: num. of records containing term (n_a)
                n_a              = len( termRecordSubset )
                intersectionSet  = list( set(termRecordSubset) & set(classLbelRecordSubset) )
                # n_ab: num. of records containing term and label (n_ab)
                n_ab             = len( intersectionSet )
                # Compute Chi-square
                result         = calcChiSquare(n_a,n_b,n_ab,len(documentTuples));
                # Compute Dice's Coefficient
                #result           = calcDiceCoeff(n_a,n_b,n_ab);
                if termRankingsPerClass.has_key(classLabel,):
                    termRankingsPerClass[classLabel].append( (term,result) );
                else:
                    termRankingsPerClass[classLabel] = [(term,result)];

        # return a list of each class label's term ranks
        # sorted in descending order
        for key,val in termRankingsPerClass.iteritems():
            tmpList = [[tuple[0],tuple[1]] for tuple in val]
            termRankingsPerClass[key] = sorted(tmpList, key=operator.itemgetter(1),reverse=True);
        return termRankingsPerClass;
    except:
        print 'error getting TermRanksPerClass'

def getRecordTermRankScoreVector(record,termRankReference,classLabel):
    try:
        recordScoreVector = [];
        # convert term rank reference to lookup
        tmpLookup = {};
        if termRankReference.has_key(classLabel):
            for t in termRankReference[classLabel]:
                tmpLookup[t[0]] = t[1];
        for term in record[2].split(' '):
            if tmpLookup.has_key(term):
                recordScoreVector.append(tmpLookup[term]);
        return recordScoreVector;
    except:
        print "error getting record score"
        
def getEncodedRecordsSubsetsForClassLabels(recordTuples,classificationLabels,termRankings):
    '''Generate record tuple subsets per classification label 
        with term feature vectors encoded with corresponding term rank scores'''
    try:
        # for each classification label
        labelRecordSubSets = {}
        for classLabel in classificationLabels:
            labelRecordSubSets[classLabel] = [];
            classLbelRecordSubset          = [t for t in recordTuples if t[1] == classLabel];
            for record in classLbelRecordSubset:
                featureVectorScore = getRecordTermRankScoreVector(record,termRankings,classLabel)
                encodedRecord      = (record[0],record[1],featureVectorScore)
                labelRecordSubSets[classLabel].append(encodedRecord)
        return labelRecordSubSets;
    except:
        print "error generating encoded record subsets for class labels"

def getStagingSamples(encodedRecordSubsets,classificationLabels,K):
    '''Generate a staging pool of K sample from each class label
        subset.'''
    try:
        stagingSample = [];
        for classLabel in classificationLabels:
            if encodedRecordSubsets.has_key(classLabel):
                tmp = random_subset( encodedRecordSubsets[classLabel], K );
                stagingSample.append( random_subset( encodedRecordSubsets[classLabel], K ) )
        # return a flattened list of records sampled from each class
        return [value for row in stagingSample for value in row];
    except:
        print "error getting staging sample"
    
def storeObject(objectToPersist,path):
    try:
        pickle.dump( objectToPersist, open( path, "wb" ) )
    except:
        print "error storing object file to disk"
    finally:
        return;

def loadObject(path):
    try:
        instanceCopy = pickle.load( open(path,"rb") );
    except:
        print "error retrieving object file from disk"
    finally:
        return instanceCopy;

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
                    ,help="Total neighbors to sample.");
    args = parser.parse_args();
    # check arguments, training set is required.
    if os.path.isfile( args.trainingSet ) is False:
        parser.print_help()
    else:
        ###Module
        # Collect required components
        recordTuples                          = getTrainingSetTuples(args.trainingSet);
        recordTuples                          = removeNumbersAndPunctuation(recordTuples);                               # Process record tuples
        classificationLabels                  = ['SMOKER','NON-SMOKER','UNKNOWN']                                          # classificiation labels
        uniqueTermsList                       = getUniqueTerms(recordTuples);                                            # get unique terms in problem space
        
        ###Module
        termRankings = [];
        # Check location of relevance labels (parameterize after testing)
        if os.path.isfile("C:\\temp\\datasets\\traningSetRankings.p"):
            # load term rankings to memory
            termRankings = loadObject(r'C:\temp\datasets\traningSetRankings.p');
        else:
            # Get term rankings per class from training set
            # Need:
            # uniqueTermsList      - for reference
            # classificationLabels - for reference
            # documentTuples       - training set to rank
            termRanksPerClass = getTermRanksPerClass(uniqueTermsList , classificationLabels, recordTuples);
            # persist rankings for training
            storeObject(termRanksPerClass,"C:\\temp\\datasets\\traningSetRankings.p");

        # Generate feature vector encoded record subsets for each class
        encodedRecordSubsets = getEncodedRecordsSubsetsForClassLabels(recordTuples,classificationLabels,termRankings)

        # get C = |c| = total number of terms that occur in training documents with class label 'c'
        totalTermsInTrainingDocPerClass       =  getTotalNumTermsInTrainingDocPerClass(recordTuples,classificationLabels)# Get total 'N'
        totalNtrainingInstances               = len(recordTuples);
        counts                                = { 'SMOKER'    : len([x for x in recordTuples if x[1] == 'SMOKER'])
                                                , 'NON-SMOKER': len([x for x in recordTuples if x[1] == 'NON-SMOKER' ])
                                                , 'UNKNOWN'   : len([x for x in recordTuples if x[1] == 'UNKNOWN']) 
                                                }
        # proportion 'P(c)' lookup of training instances with label 'c'
        proportionLookup = {
                             'SMOKER'    : float(counts['SMOKER'])     / float(totalNtrainingInstances)
                            ,'NON-SMOKER': float(counts['NON-SMOKER']) / float(totalNtrainingInstances)
                            ,'UNKNOWN'   : float(counts['UNKNOWN'])    / float(totalNtrainingInstances)
                            };
        ###Module
        # For each document in the training set, compute similarity to 'K' nearest neighbors for each label
        K = args.kNeighbors
        stagingCollection  = getStagingSamples(encodedRecordSubsets,classificationLabels,K)
        # compute similarity vectors for each class label for each record
        # for each document in the training set...
        recordClassificationStats = {};
        for record in recordTuples:
            recordId                                 = record[0]
            recordLabel                              = record[1]
            recordTermFeatureVector                  = record[2].split(' ');
            recordClassificationStats[recordId]      = {'distance':{} , 'actual_label':recordLabel};
            # compute 'K' nearest neighbor
            # get k-random neighbors
            sample = random_subset( stagingCollection, K )
            for kthNeighbor in sample:
                kthNeighborId                              = kthNeighbor[0];
                neighborTermFeatureVector                  = kthNeighbor[2];
                # get this examples vector score for this neighbors SMOKING STATUS class label
                recordTermFeatureVector                    = getRecordTermRankScoreVector(record,termRankings,kthNeighbor[1]);
                # Calculate the similarity score
                similarityScore = calculateVectorSimilarity(neighborTermFeatureVector, recordTermFeatureVector);
                recordClassificationStats[recordId]['distance'][kthNeighborId]     = similarityScore;
        
        ###Module
        # get minimum distance from each neighbor
        finalLabels = {};
        for vector in recordClassificationStats.iteritems():
            # identify which neighbor was 'nearest' to
            # each example and output the nearest neighbor's actual class along with
            # the example's predicted class.
            distanceFromNeighbor      = vector[1]['distance']
            maxProb                   = min(distanceFromNeighbor.iteritems(), key=operator.itemgetter(1))[0]
            predictedLabel = '';
            for doc in recordTuples:
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
