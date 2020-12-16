#Author: A0213047R

import re
import nltk
import sys
import getopt
import pickle
import math
from collections import Counter
from collections import defaultdict
from nltk.corpus import wordnet

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dictionary_file, postings_file, queries_file, results_file):
    """
    From the dictionary file and postings file, processes a singale query from
    from the query file, supports free text, phrases, and boolean queryies.
    Outputs all results to the results file using lnc.ltc weighting.
    """

    print('running search on the queries...')

    stemmer = nltk.stem.porter.PorterStemmer()

    results_file = open(results_file, 'w')
    post_file = open(postings_file, 'rb')
    dict_file = open(dictionary_file, 'rb')
    vocab = pickle.load(dict_file)
    docNormals = pickle.load(dict_file)
    collectionSize = pickle.load(dict_file)

    with open(queries_file, 'r') as queries_file:
        query = queries_file.readline()
        queryNorm = 0
        inPhrase = False
        # strict is used to indicate a boolean retrieval search
        strict = False
        if('AND' in query.split()): strict = True
        query = query.lower()
        query = nltk.word_tokenize(query)

        tempPhrases = []
        stemmedQuery = []
        singlePhrase = []
        for term in query:
            # `` is the start quotations of a phrase
            if(term == '``'):
                inPhrase = True
            # '' is the end quotations of a phrase
            elif(term == '\'\''):
                inPhrase = False
                if len(singlePhrase) > 0:
                    tempPhrases.append(singlePhrase)
                singlePhrase = []
            elif(inPhrase):
                singlePhrase.append(stemmer.stem(term))
                if(strict == False):
                    try:
                        syns = wordnet.synsets(term)
                        topSyn = syns[0].lemmas()[0].name()
                        stemmedQuery.append(stemmer.stem(topSyn))
                        # secondSyn = syns[1].lemmas()[0].name()
                        # stemmedQuery.append(stemmer.stem(secondSyn))
                    except IndexError:
                        pass
            else:
                stemmedQuery.append(stemmer.stem(term))
                if(strict == False):
                    try:
                        syns = wordnet.synsets(term)
                        topSyn = syns[0].lemmas()[0].name()
                        stemmedQuery.append(stemmer.stem(topSyn))
                        # secondSyn = syns[1].lemmas()[0].name()
                        # stemmedQuery.append(stemmer.stem(secondSyn))
                    except IndexError:
                        pass

        querySingleTerms = Counter(stemmedQuery)
        phrasePostings = defaultdict(list)
        phrases = []
        # counts the number of times a phrase appears in a query for scoring
        for i in range(len(tempPhrases)):
            if tempPhrases[i] not in phrases:
                count = 1
                for j in range(i+1, len(tempPhrases)):
                     if tempPhrases[i] == tempPhrases[j]:
                         count += 1
                phrases.append(tempPhrases[i])
                phrasePostings[len(phrases)-1].append(count)

        if(len(phrases) > 0):
            for p in range(len(phrases)):
                postings = []
                skip = False
                for word in phrases[p]:
                    try:
                        _, index = vocab[word]
                        post_file.seek(index)
                        postings.append(pickle.load(post_file))
                    except KeyError:
                        # if a word from a phrase is not found, no docs will match that phrase
                        phrasePostings[p] = []
                        skip = True
                        break

                if(skip != True):

                    interPostings = []
                    if(len(postings) > 1):
                        # finds the common docs
                        interDocs = intersection(postings[0], postings[1], 3, 3)
                        for doc in interDocs:
                            # finds the adjacent postitions in common docs
                            interPositions = intersection(postings[0][postings[0].index(doc)+2], postings[1][postings[1].index(doc)+2], 1, 1, 1)
                            if(len(interPositions) > 0):
                                # Checking whether there is a third term and need to merge postings again
                                if(len(postings) > 2):
                                    interPostings.append(doc)
                                    interPostings.append(interPositions)
                                else:
                                    phrasePostings[p].append(doc)
                                    phrasePostings[p].append(len(interPositions))
                    else:
                        for i in range(0, len(postings[0]), 3):
                            phrasePostings[p].append(postings[0][i])
                            phrasePostings[p].append(postings[0][i+1])

                    if(len(postings) > 2):
                        interDocs = intersection(interPostings, postings[2], 2, 3)
                        for doc in interDocs:
                            interPositions = intersection(interPostings[interPostings.index(doc)+1], postings[2][postings[2].index(doc)+2], 1, 1, 1)
                            if(len(interPositions) > 0):
                                phrasePostings[p].append(doc)
                                phrasePostings[p].append(len(interPositions))

        strictPostings = []
        if(strict):
            postings = []
            for word in querySingleTerms.keys():
                try:
                    _, index = vocab[word]
                    post_file.seek(index)
                    post = pickle.load(post_file)
                    postings.append((len(post)/3, post))
                except KeyError:
                    # If the word isn't in any document there are 0 matches, so stop here
                    results_file.write("\n")
                    return
            # sorts based on length of the postings lists
            postings.sort()
            # Makes strictPostings initially equal to the smallest postings list
            for docId in postings[0][1][::3]:
                strictPostings.append(docId)
            # strictly merges all postings lists
            for _, post in postings[1:]:
                tempPost = intersection(strictPostings, post, 1, 3)
                strictPostings = tempPost
            # strictly merges single term postings with phrase postings
            for phrasePost in phrasePostings.values():
                tempPost = intersection(strictPostings, phrasePost[1:], 1, 2)
                strictPostings = tempPost

        docScores = defaultdict(list)
        queryScores = {}
        for word, wordCount in querySingleTerms.items():
            try:
                docFreq, index = vocab[word]
                post_file.seek(index)
                postings = pickle.load(post_file)
                qidf = math.log(collectionSize / docFreq, 10)
            except KeyError:
                docFreq = 0
                postings = []
                qidf = 0

            qtf = 1 + math.log(wordCount, 10)
            qwt = qtf * qidf
            queryNorm += qwt ** 2
            queryScores[word] = qwt

            # Iterates over postings list calculating dtf for each doc
            for i in range(0, len(postings), 3):
                if(strict == False or postings[i] in strictPostings):
                    dtf = 1 + math.log(postings[i+1], 10)
                    docScores[postings[i]].append((dtf, word))

        for phraseNum, postings in phrasePostings.items():
            try:
                qidf = math.log(collectionSize / ((len(postings)-1) / 2), 10)
            except ZeroDivisionError:
                qidf = 0
            qtf = 1 + math.log(postings[0], 10)
            qwt = qtf * qidf
            queryNorm += qwt ** 2
            queryScores[phraseNum] = qwt

            for i in range(1, len(postings), 2):
                if(strict == False or postings[i] in strictPostings):
                    dtf = 1 + math.log(postings[i+1], 10)
                    docScores[postings[i]].append((dtf, phraseNum))

        # Normalize the query scores
        if queryNorm != 0:
            queryNorm = 1 / math.sqrt(queryNorm)
        for word, value in queryScores.items():
            queryScores[word] = value * queryNorm

        # Normalize the doc scores, then calculates the product of the
        # doc and query score for each word, then sums these for final score
        for doc, dScoreVec in docScores.items():
            finalScore = 0
            docNorm = docNormals[doc]
            for dScore in dScoreVec:
                score = dScore[0] * docNorm
                finalScore += score * queryScores[dScore[1]]
            docScores[doc] = finalScore

        for docId, score in sorted(docScores.items(), key=lambda item: item[1], reverse = True):
            results_file.write(str(docId) + " ")
        results_file.write("\n")

    results_file.close()
    post_file.close()
    dict_file.close()

def intersection(a, b, aInc, bInc, phrase = 0):
    '''
    Finds the common terms in lists a and b, incrementing by aInc and bInc.
    phrase is to indicate whether we are searching for a phrase, in which case
    a and b are postition lists and we should store results that differ by 1.
    '''

    result = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        # Need to convert from str to int to apply the 1 position difference for phrases
        if(int(a[i]) + phrase == int(b[j])):
            result.append(b[j])
            i += aInc
            j += bInc
        elif a[i] < b[j]:
            i += aInc
        else:
            j += bInc
    return result

dictionary_file = postings_file = file_of_queries = file_of_output = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
