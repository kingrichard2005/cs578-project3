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

def hamdist(str1, str2):
    '''Count the # of differences between equal length 
    strings str1 and str2, borrowed from http://code.activestate.com/recipes/499304-hamming-distance/'''
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs;

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
            content = content.replace("<ROOT>\n","");
            content = content.replace("</ROOT>\n","");
            content    = content.split("\n")
            content    = [m for m in content]
            tmpStr     = ''
            tmpStrList = []
            for m in content:
                if m == '':
                    tmpStrList.append(tmpStr);
                    tmpStr = ''
                else:
                    tmpStr += ' {0}'.format(m);

            # Extract tuples from parsed records
            pattern   = re.compile(R'<RECORD\sID="(\d*)">\s*<SMOKING\sSTATUS="(.*)"></SMOKING>\s*<TEXT>\s*([^<>]+)\s*</TEXT>\s*(?:</RECORD>)');
            recordIdTupleList = [];
            #with open(R"C:\temp\datasets\knnDiagnosticsParsedRecords.txt", "a") as diag_file:
            for str in tmpStrList:
                #diag_file.writelines(str);
                recordIdTupleList.append([m for m in pattern.findall(str)]);
            recordIdTupleList = [t for t in recordIdTupleList if len(t) != 0];
        return recordIdTupleList;
    except:
        print "error getting record tuples from training set file {0}".format(str(trainingSet));

def calculateVectorSimilarity(x, y, similarity_func = 'euclidean'):
    '''Calculate the similarity between two vectors using the Euclidean Distance function.'''
    try:
        similarityDistance = 0.0;
        localX = x[:];
        localY = y[:];
        sumAcc = 0;
        # if lengths differ
        # pad shorter vector with zeroes
        if abs(len(x)-len(y)) != 0:
            zeroPad = [0] * abs(len(x)-len(y));
            if ( len(x) < len(y) ):
                for e in zeroPad:
                    localX.append(e);
            else:
                for e in zeroPad:
                    localY.append(e);
        
        sumAcc += len(localY)

        if similarity_func == 'minkowski':
            for a in xrange(0,sumAcc):
                p = localX[a];
                similarityDistance += math.pow( float( abs(localX[a]-localY[a]) ),float(p) );
            similarityDistance = math.pow( similarityDistance,float( 1/len(localX) ) );
        elif similarity_func == 'manhatten':
            for a in xrange(0,sumAcc):
                similarityDistance += float( abs(localX[a]-localY[a]) );
            similarityDistance = math.sqrt(similarityDistance);
        # default to Euclidean distance measure
        else:
            for a in xrange(0,sumAcc):
                similarityDistance += float( (localX[a]-localY[a])**2 );
            similarityDistance = math.sqrt(similarityDistance);
        
        # return distance measure rounded to 4 decimal places
        return round(similarityDistance,4);
    except:
        print "error calculating similarity score"

def calcFeatureHammingDistance(a, b):
    '''Calculate the Hamming distance of the binary representation of term string elements 
        from equal length vectors.  The longer vector is trimmed to the length of the shorter vector.'''
    try:
        FeatureHammingDistance = 0.0;
        commonTerms = list(set(a) & set(b));
        for x, y in zip(a, b):
            # terms as a list of binary encoded strings
            list1 = ' '.join(format(ord(xF), 'b') for xF in x).split(' ')
            list2 = ' '.join(format(ord(yF), 'b') for yF in y).split(' ')
            # match list lengths by setting lists equal to shortest list
            # NOTE:  Review since we are essentially trimming extra terms
            # from the record with the larger term feature vector
            list1 = list1[0:len(list2)] if ( len(list2) < len(list1) ) else list1
            list2 = list2[0:len(list1)] if ( len(list1) < len(list2) ) else list2
            for subx, suby in zip(list1, list2):
                FeatureHammingDistance += hamdist(subx, suby)

        return FeatureHammingDistance;
    except:
        print "Error calculating hamming distance for unlabeled example and kth-neighbor "

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

def getTermRanksPerClass(uniqueTermsList , classificationLabels, documentTuples, associationFunction = 'chi-sq'):
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
                # Compute using Chi-square ( or Dice's Coefficient )
                result         = calcChiSquare(n_a,n_b,n_ab,len(documentTuples)) if associationFunction == 'chi-sq' else calcDiceCoeff(n_a,n_b,n_ab);
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
        
def getEncodedRecordsSubsetsForClassLabels(recordTuples,classificationLabels,termRankings,preserveTerms = False):
    '''Generate record tuple subsets per classification label 
        with term feature vectors encoded with corresponding term rank scores'''
    try:
        # for each classification label
        labelRecordSubSets = {}
        for classLabel in classificationLabels:
            labelRecordSubSets[classLabel] = [];
            classLbelRecordSubset          = [t for t in recordTuples if t[1] == classLabel];
            for record in classLbelRecordSubset:
                featureVectorScore =  [] if preserveTerms else getRecordTermRankScoreVector(record,termRankings,classLabel);
                encodedRecord      = (record[0],record[1], record[2] if preserveTerms else featureVectorScore )
                labelRecordSubSets[classLabel].append(encodedRecord)
        return labelRecordSubSets;
    except:
        print "error generating encoded record subsets for class labels"

def getTopKScoringVectors(collection,K):
    '''Get top K scoring records from collection'''
    try:
        topScoring  = []
        topKRecords = []
        # TODO: revise
        for record in collection:
            # sum score
            topScoring.append([record[0],sum(record[2])])
        topScoring = sorted(topScoring, key=operator.itemgetter(1),reverse=True);

        for topRecord in topScoring[0:K]:
            for record in collection:
                if record[0] == topRecord[0]:
                    topKRecords.append(record);
        return topKRecords;
    except:
        print "error getting top K vectors"

def getStagingSamples(encodedRecordSubsets,classificationLabels,K, sampleType = 'Krandom', preserveTerms = False):
    '''Generate a staging pool of K sample from each class label
        subset.'''
    try:
        stagingSample = [];
        for classLabel in classificationLabels:
            if encodedRecordSubsets.has_key(classLabel):
                # get get 'K' top-scoring records from each class label subset
                subset = encodedRecordSubsets[classLabel];
                if sampleType == 'topK' and preserveTerms == False:
                    tmp    = getTopKScoringVectors(subset,K);
                # default k-random samples
                else:
                    tmp    = random_subset( subset, K );
                stagingSample.append( tmp )
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

def listToChunks(list , N):
    '''Use a Python generator to split a list
        to N size chunks.'''
    try:
        for i in xrange(0, len(list), N):
            # Similar to 'take', on each successive
            # iteration will 'yield' as much of the list
            # up to 'N' for len(list) > N length, else when 
            # len(list) < N, just return remaining list elements.
            yield list[i:i+N];
    except:
        print "Error splitting list into {0} chunks".format(str(N));

def getClusterPrecision(recordTuples, classificationLabels, K):
    '''Calculate the cluster precision for K clusters derived
        from the input training record set. 
        ( see Search Engines: Information Retrieval in Practice 9.2.3 )'''
    try:
        # Split the record set into 'K' clusters
        kChunk             = (len(recordTuples) / K) + 1;
        kClusters          = list( listToChunks(recordTuples, kChunk) );
        # sum the max values
        sumMax  = 0;
        for classLabel in classificationLabels:
            labelCountAcc = [];
            for cluster in kClusters:
                labelCountAcc.append(len([ e for e in cluster if e[1] == classLabel ]));
            sumMax  += max( labelCountAcc );

        return round( abs(sumMax) / float(K),3);
    except:
        print "error getting cluster precision"

if __name__ == '__main__':
    '''Train a Naive Bayes Classifier to classify 
        a set of (scrubbed) medical records with the label(s) 
        'SMOKER','NON-SMOKER' or 'UNKNOWN' '''
    parser = argparse.ArgumentParser(prog="K-Nearest Neighbor Classifier"
                                     ,description='Trains a K-Nearest Neighbor classifier '\
                                         'to label patients as SMOKING, NON-SMOKING or \
                                         UNKNOWN based on available (scrubbed) medical record information.')
    parser.add_argument('-t', '--trainingSet'
                        ,help='Path to the labeled medical record training set \
                        file, e.g. ./path/to/training.txt'
                        ,default=''
                        ,dest="trainingSet");
    parser.add_argument('-r', '--termrankings'
                        ,help='Path to term rankings pickle file, e.g. ./path/to/termRankings.p'
                        ,default=''
                        ,dest="termRankings");
    parser.add_argument('-a', '--associationFunction'
                        ,help='The association function used to compare the relevancy of a \
                        term to a specific class label [default=chi-square|dice].'
                        ,default='chi-sq'
                        ,dest="assocFunc");
    parser.add_argument('-K','--kNeighbors'
                    ,dest="kNeighbors"
                    ,default = 3
                    ,type=int
                    ,help="Total neighbors to sample.");
    parser.add_argument('-s','--similarity'
                    ,dest="similarity_func"
                    ,default = 'euclidean'
                    ,help="The similarity function used to compare a unlabeled examples \
                    to a labeled kth-neighbor [default=euclidean|manhatten|minkowski].");
    parser.add_argument('-S','--sampleType'
                    ,dest="sampleType"
                    ,default = 'Krandom'
                    ,help="The method to sample 'K' records from each label subset, top 'K''\
                    are records with the max combined term relevance score [default=Krandom|topK],'\
                    this sample type doesn't apply when using the hamming distance similarity.");

    args = parser.parse_args();
    # check arguments, training set is required.
    if os.path.isfile( args.trainingSet ) is False or os.path.isfile( args.termRankings ) is False:
        parser.print_help()
    else:
        ###Module

        # Collect required components
        recordTuples                          = getTrainingSetTuples(args.trainingSet);
        recordTuples                          = removeNumbersAndPunctuation(recordTuples);     # Process record tuples
        classificationLabels                  = ['SMOKER','NON-SMOKER','UNKNOWN']# classificiation labels
        uniqueTermsList                       = getUniqueTerms(recordTuples);                  # get unique terms in problem space
        similarityFunction                    = str.strip(args.similarity_func);
        sampleType                            = str.strip(args.sampleType);
        preserveTerms                         = True if similarityFunction == 'hamming' else False;
        associationFunction                   = args.assocFunc;
        
        ###Module
        termRankings = [];
        # Check location of relevance labels (parameterize after testing)
        if os.path.isfile(args.termRankings):
            # load term rankings to memory
            termRankings = loadObject(args.termRankings);
        else:
            # Get term rankings per class from training set
            # Need:
            # uniqueTermsList      - for reference
            # classificationLabels - for reference
            # documentTuples       - training set to rank
            termRanksPerClass = getTermRanksPerClass(uniqueTermsList , classificationLabels, recordTuples, associationFunction);
            # persist rankings for training
            storeObject(termRanksPerClass,args.termRankings);

        print "Cluster precision with '{0}'-neighbors estimated at {1}%".format(args.kNeighbors,str(getClusterPrecision(recordTuples, classificationLabels, args.kNeighbors)))        
        # Generate feature vector encoded record subsets for each class
        encodedRecordSubsets = getEncodedRecordsSubsetsForClassLabels(recordTuples,classificationLabels,termRankings, preserveTerms)

        ###Module
        # For each document in the training set, compute similarity to 'K' nearest neighbors for each label
        K = args.kNeighbors
        stagingCollection  = getStagingSamples(encodedRecordSubsets,classificationLabels,K,sampleType, preserveTerms)
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
                # get this examples vector score if we're not calculating hamming distance
                recordTermFeatureVector                    = recordTermFeatureVector if preserveTerms else getRecordTermRankScoreVector(record,termRankings,kthNeighbor[1]);
                # Calculate the similarity score
                similarityScore = calcFeatureHammingDistance(neighborTermFeatureVector, recordTermFeatureVector) if similarityFunction == 'hamming' else calculateVectorSimilarity(neighborTermFeatureVector, recordTermFeatureVector, similarityFunction);
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
            predictedLabel            = '';
            for doc in recordTuples:
                if doc[0] == maxProb:
                    predictedLabel = doc[1];
                    break;
            finalLabels[vector[0]]    = [predictedLabel,vector[1]['actual_label']];

        print "'{0}' records classified".format(len(finalLabels))
        actualLbls        = [];
        predictedLbls     = [];
        longestClassLabel = max(len(s) for s in classificationLabels)
        for output in sorted(finalLabels.iteritems(), key=operator.itemgetter(0)):
            tmp      = [" "] * abs( len(output[1][0]) - longestClassLabel);
            spacePad = "".join(tmp);
            actualLbls.append( output[1][1] );
            if output[1][0] == output[1][1]:
                predictedLbls.append( output[1][0] );
            print "Record Id: '{0:03}' Classified as '{1}'{2},Actual Classification: '{3}'".format( int(output[0]) ,output[1][0], spacePad,output[1][1]);
        print "Correctly classified: '{0}' of '{1}' records".format( len(predictedLbls),len(actualLbls) );