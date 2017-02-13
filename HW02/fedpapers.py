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
#vectorizer.vocabulary_

# Split word counts into separate matrices
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]


# Initialize vectors for P(word_j | H/M) as total occurence of a word for an other divided by total words 
fH = np.zeros(len(XH[0]))
totH = 0
fM = np.zeros(len(XM[0]))
totM = 0

# Estimate probability of each word in vocabulary being used by Hamilton
for i in range(0,len(XH[0])):
    for j in range(0,len(XH)):
        fH[i] = float(fH[i])+XH[j][i]
    totH = totH + fH[i]
fH = fH/totH

# Estimate probability of each word in vocabulary being used by Madison
for i in range(0,len(XM[0])):
    for j in range(0,len(XM)):
        fM[i] = float(fM[i])+XM[j][i]
    totM = totM + fM[i]
fM = fM/totM

# Compute ratio of these probabilities
fratio = fH/fM

# Compute prior probabilities 
piH = float(nH)/(nH+nM)
piM = float(nM)/(nH+nM)

Ham_tot = nH
Mad_tot = nM
for xd in XD: # Iterate over disputed documents
    rat = 1
    # Compute likelihood ratio for Naive Bayes model
    for j in range(0,len(xd)):
        rat = rat*(fratio[j])**xd[j]
        
    LR = (piH/piM)*rat
    if LR>1:
        print 'Hamilton'
        Ham_tot=Ham_tot+1
    else:
        print 'Madison'
        Mad_tot=Mad_tot+1
print "Hamilton wrote %d total, and %d of the disputed." % (Ham_tot, Ham_tot-nH)
print "Madison wrote %d total, and %d of the disputed." % (Mad_tot, Mad_tot-nM)
    