#Author: A0213047R

import re
import nltk
import sys
import getopt
import pickle
import os
import math
import csv

# fixes issue with CSV size limit
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    """
    Builds index from the CSV file found in the input directory,
    then output the dictionary file and postings file. Also
    calculates the number for normalization for each document
    and stores it in the dictionary file.
    """

    print('indexing...')

    vocab = {}
    docSizes = {}
    docNormals = {}
    stemmer = nltk.stem.porter.PorterStemmer()

    collectionSize = 0

    with open(in_dir, newline = '', encoding="utf8") as file:
        data = csv.reader(file)
        next(data) # skips column headers
        for case in data:
            collectionSize += 1
            content = case[2].lower()
            sentences = nltk.sent_tokenize(content)
            normal = {}
            position = 0
            for sent in sentences:
                words = nltk.word_tokenize(sent)
                words = [stemmer.stem(term) for term in words]
                for word in words:
                    if word in vocab.keys():
                        if case[0] != vocab[word][-3]:
                            vocab[word].extend([case[0], 1, [position]])
                            normal[word] = 1
                        else:
                            vocab[word][-2] += 1
                            vocab[word][-1].append(position)
                            # sometimes the case number appears twice in a row
                            if(word not in normal.keys()):
                                normal[word] = 1
                            else:
                                normal[word] += 1
                    else:
                        vocab[word] = [case[0], 1, [position]]
                        normal[word] = 1
                    position += 1
            # Calculates the number to normalize document score vectors
            normalize = 0
            for num in normal.values():
                normalize += num ** 2
            normalize = 1 / math.sqrt(normalize)
            docNormals[case[0]] = normalize

    post_file = open(out_postings, 'wb')
    dict_file = open(out_dict, 'wb')
    index_dict = {}

    for word, postings in vocab.items():
        index = post_file.tell()
        pickle.dump(postings, post_file, 1)
        docFreq = len(postings) / 3
        index_dict[word] = docFreq, index
        
    pickle.dump(index_dict, dict_file)
    pickle.dump(docNormals, dict_file)
    pickle.dump(collectionSize, dict_file)

    post_file.close()
    dict_file.close()

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
