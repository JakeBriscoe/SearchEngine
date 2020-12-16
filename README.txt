This is the README file for A0213047R's submission
e0505830@u.nus.edu

== Python Version ==

I'm using Python Version 3.8

== General Notes about this assignment ==

Indexing:
To index the documents in the given directory I process the documents in order,
performing sentence and word tokenization on the contents of each file, also
converting to lower case and stemming the word, no other word processing is
applied. There is a dictionary with the keys being all unique words found, and the
corresponding values being the document IDs that the words were found in, and the
count of that word in that document, and a list of the positions that word was
found in that document. The postings lists are then dumped individually
to the postings file using pickle, with the indexes for the postings lists and
corresponding words then being dumped to the dictionary file, this also includes
the document frequency for the word. The number required to normalize each document
score vector is also calculated and put in a dictionary with the key being the
document ID, this is also written to the dictionary file.

Searching:
The query terms are initially processed to find whether it is a boolean search,
referred to as strict in the code, and separates individual words and phrases.
If the query is not boolean then each word in the query has one synonym added
as free text to the query, I experimented with two synonyms being added and got
very slightly higher results but I think the query can lose some meaning with
more non relevant synonyms being added, so I have kept it at one.
A postings list for each phrase is then found of all docs where each word in the
phrase appear one after the other in order in that doc. Then if it is a strict
boolean search a common docs postings list is found that matches each word and
phrase in the query, when calculating document scores only the documents found
in the common postings list are used.
The scores for the query and documents are then calculated using the lnc.ltc scheme
After this is done the query score vector is normalized, as is each document
score vector from the number calculated during indexing. The angle between these
two is then calculated all results in order are written to the results file.

== Files included with this submission ==

index.py - Indexes the documents
search.py - Performs queries on the documents
dictionary.txt - Contains each word in the collection
postings.txt - Contains the documents each word appears in

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I, A0213047R, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

== References ==

I looked at the python docs, geeks for geeks and stackoverflow for
clarification on Python/how to implement things
https://docs.python.org/3/
https://stackoverflow.com/
https://www.geeksforgeeks.org/

How to process CSV files
https://docs.python.org/3/library/csv.html

I directly copied code on how to fix a CSV field size error from this link
https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
