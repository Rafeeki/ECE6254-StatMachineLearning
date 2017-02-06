import numpy as np
import json
from sklearn.feature_extraction import text

x = open('fedpapers_split.txt').read()
papers = json.loads(x)

papersH = papers[0] # papers by Hamilton 
papersM = papers[1] # papers by Madison
papersD = papers[2] # disputed papers

nH, nM, nD = len(papersH), len(papersM), len(papersD)

# This allows you to ignore certain common words in English
# You may want to experiment by choosing the second option or your own
# list of stop words, but be sure to keep 'HAMILTON' and 'MADISON' in
# this list at a minimum, as their names appear in the text of the papers
# and leaving them in could lead to unpredictable results
stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON','MADISON'})
#stop_words = {'HAMILTON','MADISON'}

## Form bag of words model using words used at least 10 times
vectorizer = text.CountVectorizer(stop_words,min_df=10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

# Uncomment this line to see the full list of words remaining after filtering out 
# stop words and words used less than min_df times
vectorizer.vocabulary_

# Split word counts into separate matrices
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]

# Estimate probability of each word in vocabulary being used by Hamilton
fH = .6 # refer to lecture #4 pages 11-15

# Estimate probability of each word in vocabulary being used by Madison
fM = .4 # refer to lecture #4 pages 11-15

# Compute ratio of these probabilities
fratio = fH/fM

# Compute prior probabilities 
piH = ???
piM = ???

for xd in XD: # Iterate over disputed documents
    # Compute likelihood ratio for Naive Bayes model
    LR = ???
    if LR>0.5:
        print 'Hamilton'
    else:
        print 'Madison'
    