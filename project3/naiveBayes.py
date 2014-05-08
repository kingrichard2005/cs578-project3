#-------------------------------------------------------------------------------
# Name:        cs-578-project 3
# Purpose:      A prototype trainer for a Naive Bayes classifier that 
# attempts to predict the smoking status likelihood for unlabeled patients 
# falling into one of three categories: 'SMOKER, 'NON-SMOKER'
# , or 'UNKNOWN'.  The classifier is trained using a precompiled collection of labeled training record data
#   
# Author:      kingrichard2005
#
# Created:     20141604
# Copyright:   (c) kingrichard2005 2014
# Licence:     MIT
#-------------------------------------------------------------------------------
import re
from bs4 import BeautifulSoup
import urllib2
import math
import operator
import os
import argparse
import string
import time

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

def getTermClassOccurenceLookup(documentTuples,uniqueTermsList, classificationLabels):
    '''Gets the number of times the a term occurs in class 'c' of each document'''
    try:
        termClassOccurenceLookup = {};
        for t in documentTuples:
            # get content as list
            contentTermList = t[2].split(' ');
            # intersect to get unique terms in this document
            uqTermsInDocument = list(set(contentTermList) & set(uniqueTermsList));
            for uq in uqTermsInDocument:
                smokingStatusIdTuple = (t[1],1)
                if termClassOccurenceLookup.has_key(uq) is False:
                    termClassOccurenceLookup[uq] = {};
                    for classLabel in classificationLabels:
                        termClassOccurenceLookup[uq][classLabel]= 0;

                if smokingStatusIdTuple[0] in classificationLabels:
                    termClassOccurenceLookup[uq][smokingStatusIdTuple[0]] += 1;
        return termClassOccurenceLookup;
    except:
        print "Error geting term class occurrence lookup"

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


def getTestSetTuples(testSet):
    """"Each training example is represented as a 
        3-tuple with the following schema: 
        ( {Record ID}, {Training Label|default=''}, {Feature Term String - used for feature vector extraction} )."""
    try:
        recordIdTupleList = [];
        with open(testSet, 'r') as content_file:
            # Read records into list
            content    = content_file.read()
            content    = content.replace("<ROOT>\n","");
            content    = content.replace("</ROOT>\n","");
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
            pattern   = re.compile(R'<RECORD\sID="(\d*)">\s*<TEXT>\s*([^<>]+)\s*</TEXT>\s*(?:</RECORD>)');
            recordIdTupleList = [];
            #with open(R"C:\temp\datasets\knnDiagnosticsParsedRecords.txt", "a") as diag_file:
            for str in tmpStrList:
                #diag_file.writelines(str);
                matchCollection = pattern.findall(str)
                recordTuple     = (matchCollection[0][0],'',matchCollection[0][1]);
                recordIdTupleList.append([recordTuple]);
            recordIdTupleList = [t for t in recordIdTupleList if len(t) != 0];
        return recordIdTupleList;
    except:
        print "error getting test set tuples"

def computePrediction( w,c,termClassOccurrenceLookup,totalTermsInTrainingDocPerClass,recordTermFeatureVector, mu = 0.75, useBayesianSmoothing = False ):
    try:
        ''' Computes the probability that term 'w' belongs to classl 'c' '''
        tf         = termClassOccurrenceLookup[w][c] if termClassOccurrenceLookup.has_key(w) else 0.0;
        cfw        = sum( i[1] for i in termClassOccurrenceLookup[w].iteritems()) if termClassOccurrenceLookup.has_key(w) else 0.0;
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
    except:
        print "error computing likelihood probability"

def generateLabeledRecordPredictions(labeledTrainingSet,classificationLabels,proportionLookup,mu, smoothingFunction = 'bayes'):
    try:
        # compute probability vectors P(C|D) for each class label for each record
        # for each document in the training set...
        recordClassificationStats = {};
        for record in labeledTrainingSet:
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
                    P_wc = computePrediction( term
                                                  ,classLabel
                                                  ,termClassOccurrenceLookup
                                                  ,totalTermsInTrainingDocPerClass
                                                  ,recordTermFeatureVector
                                                  ,mu
                                                  ,True if smoothingFunction == 'bayes' else False #useBayesianSmoothing = args.BayesianSmoothing 
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

        print "...'{0}' records classified\n".format(len(finalLabels))
        actualLbls        = [];
        predictedLbls     = [];
        longestClassLabel = max(len(s) for s in classificationLabels)
        for output in sorted(finalLabels.iteritems(), key=operator.itemgetter(0)):
            tmp      = [" "] * abs( len(output[1][0]) - longestClassLabel);
            spacePad = "".join(tmp);
            actualLbls.append( (output[0],output[1][1]) );
            if output[1][0] == output[1][1]:
                predictedLbls.append( (output[0],output[1][0]) );
            print "Record Id: '{0:03}' Classified as '{1}'{2},Actual Classification: '{3}'".format( int(output[0]) ,output[1][0], spacePad,output[1][1]);

        print "\n"
        for label in classificationLabels:
            relA                  = [ a for a in actualLbls if a[1]    == label ]
            notRelA               = [ a for a in actualLbls if a[1]    != label ]
            retB                  = [ p for p in predictedLbls if p[1] == label ]
            notRetB               = [ p for p in predictedLbls if p[1] != label ]

            intersect_RelARetB    = abs(len([i for i in  relA if i in retB]));
            precision             = float(intersect_RelARetB) / float( len(relA) ) if relA > 0 else 0;
            recall                = float(intersect_RelARetB) / float( len(retB) ) if retB > 0 else 0; 
            F1Score               = round(float(2 * (precision * recall)) / float(precision + recall), 3) if (precision + recall) > 0 else 0;
            print "Label '{0}':".format(label)
            print "Precision with mu at '{0}', {1}%".format(mu,str( round((precision * 100),3) ) )
            print "Recall with mu '{0}', {1}%".format(mu,str( round((recall * 100),3) ) )
            print "F1 score with mu '{0}', {1}\n".format(mu,str(F1Score) )

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

        return 0;
    except:
        print "error generating Labeled Record Predictions"

def generateUnlabeledRecordPredictions(UnlabeledTestSet,classificationLabels,proportionLookup,mu, smoothingFunction = 'bayes'):
    try:
        # compute probability vectors P(C|D) for each class label for each record
        # for each document in the training set...
        recordClassificationStats = {};
        for record in UnlabeledTestSet:
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
                    P_wc = computePrediction( term
                                                  ,classLabel
                                                  ,termClassOccurrenceLookup
                                                  ,totalTermsInTrainingDocPerClass
                                                  ,recordTermFeatureVector
                                                  ,mu
                                                  ,True if smoothingFunction == 'bayes' else False #useBayesianSmoothing = args.BayesianSmoothing 
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

        print "...'{0}' records classified\n".format(len(finalLabels))
        actualLbls        = [];
        predictedLbls     = [];
        longestClassLabel = max(len(s) for s in classificationLabels)
        for output in sorted(finalLabels.iteritems(), key=operator.itemgetter(0)):
            tmp      = [" "] * abs( len(output[1][0]) - longestClassLabel);
            spacePad = "".join(tmp);
            actualLbls.append( (output[0],output[1][1]) );
            if output[1][0] == output[1][1]:
                predictedLbls.append( (output[0],output[1][0]) );
            print "Record Id: '{0:03}' Classified as '{1}'{2}".format( int(output[0]) ,output[1][0], spacePad);

        actualLbls        = [];
        predictedLbls     = [];
        longestClassLabel = max(len(s) for s in classificationLabels)
        for output in sorted(finalLabels.iteritems(), key=operator.itemgetter(0)):
            tmp      = [" "] * abs( len(output[1][0]) - longestClassLabel);
            spacePad = "".join(tmp);
            actualLbls.append( output[1][1] );
            if output[1][0] == output[1][1]:
                predictedLbls.append( output[1][0] );
            print "Record Id: '{0:03}' Classified as '{1}'{2}".format( int(output[0]) ,output[1][0], spacePad);
        return 0;
    except:
        print "error generating predictions for unlabeled tests set"

def scrapeNsciaTerms(pattern):
    '''Read and parse medical terms from the National Spinal Cord Injury Association (NSCIA) Encyclopedia
        http://www.spinalcord.org/resource-center/askus/index.php?pg=kb.page&id=1413'''
    urlResp          = urllib2.urlopen(r"http://www.spinalcord.org/resource-center/askus/index.php?pg=kb.page&id=1413");
    content          = urlResp.read();
    soup             = BeautifulSoup(content)
    NsciaTermsDict   = []
    for paragraph in soup.find_all("p"):
        termTupleMatch = re.findall(R'(<em>([^<>]*)</em></strong></span>([^<>]*)<br/>)',str(paragraph));
        if len(termTupleMatch) > 0:
            for match in termTupleMatch:
                # remove non-printable char
                # TODO: update regex pattern to ignore these so this is not necessary
                cleanString = match[2].replace('\xc2\x96','').strip()
                NsciaTermsDict.append( (match[1],cleanString) );
    return NsciaTermsDict;

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
    parser.add_argument('-c', '--testSet'
                ,help='Path to the unlabeled medical record test set \
                file, e.g. ./path/to/test_set.txt'
                ,default=''
                ,dest="testSet");
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
    if os.path.isfile( args.trainingSet ) is False or os.path.isfile( args.testSet ) is False:
        parser.print_help()
    else:
        ####
        # Collect required components
        classificationLabels                  = ['SMOKER','NON-SMOKER','UNKNOWN']                                                                # classificiation labels
        labeledTrainingSet                    = getTrainingSetTuples(args.trainingSet);
        labeledTrainingSet                    = removeNumbersAndPunctuation(labeledTrainingSet);                                                     # Process record tuples
        unlabeledTestSet                      = getTestSetTuples(args.testSet);
        unlabeledTestSet                      = removeNumbersAndPunctuation(unlabeledTestSet);
        uniqueTermsList                       = getUniqueTerms(labeledTrainingSet);                                                                  # get unique terms in problem space
        termClassOccurrenceLookup             = getTermClassOccurenceLookup(labeledTrainingSet,uniqueTermsList, classificationLabels);               # get Tf_wc
        # get C = |c| = total number of terms that occur in training documents with class label 'c'
        totalTermsInTrainingDocPerClass       =  getTotalNumTermsInTrainingDocPerClass(labeledTrainingSet,classificationLabels)# Get total 'N'
        totalNtrainingInstances               = len(labeledTrainingSet);
        counts                                = { 'SMOKER'    : len([x for x in labeledTrainingSet if x[1] == 'SMOKER'])
                                                , 'NON-SMOKER': len([x for x in labeledTrainingSet if x[1] == 'NON-SMOKER' ])
                                                , 'UNKNOWN'   : len([x for x in labeledTrainingSet if x[1] == 'UNKNOWN']) 
                                                }
        # proportion 'P(c)' lookup of training instances with class label 'c'
        proportionLookup = {
                             'SMOKER'    : float(counts['SMOKER'])     / float(totalNtrainingInstances)
                            ,'NON-SMOKER': float(counts['NON-SMOKER']) / float(totalNtrainingInstances)
                            ,'UNKNOWN'   : float(counts['UNKNOWN'])    / float(totalNtrainingInstances)
                            };
        ####
        # For each document in the training set, compute probability for each label
        # configurable tuning parameter...or do we set it to the total unique terms in the record collection?
        t0                              = time.time()
        mu = args.mu if args.mu is not None else len(uniqueTermsList);
        
        generateLabeledRecordPredictions(
                                         labeledTrainingSet
                                         ,classificationLabels
                                         ,proportionLookup
                                         ,mu
                                         ,'bayes' if args.BayesianSmoothing == True else 'Dirichlet'
                                         );
        print "\nTime to Train: '{0}' seconds\n".format( str( round(time.time() - t0,4) ) );
        
        print "Classifying Test Set..\n";
        generateUnlabeledRecordPredictions(
                                         unlabeledTestSet
                                         ,classificationLabels
                                         ,proportionLookup
                                         ,mu
                                         ,'bayes' if args.BayesianSmoothing == True else 'Dirichlet'
                                         );
        print "\nClassifying Test Set..";