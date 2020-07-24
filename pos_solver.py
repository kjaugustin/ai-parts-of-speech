###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#1. Mandar Baxi msbaxi
#2. Augustine Joseph Aujoseph
#3. Milind Suryawanshi msuryawa
# (Based on skeleton code by D. Crandall)
#
#
####
# 1. train() calculates all the probabilities from the training set that
# are to be used by solver. it calculates initial probabilities, transition
# probabilities, POS tag probabilities, Emission probabilities
# 2. simplified() tags the words only using the max probability of a label 
# given the word.It is based on the simplified bias net in fig 1(b)
# 3. hmm_ve() considers model  dependencies between words. For first word,it
# considers probablility of tag given word and initial probablity of tag and 
# selects maximum of initial probabaility(tag)*probability(tag|word)
# from second word onwards, it considers probablity of tag given word and 
# probability of word given prevvoius word. It selects maximum of 
# probablilty(tag|word)*probablilty(word|previous_word)
# 4. hmm_viterbi() uses viterbi algorithm. It calculates the probability 
# of all the # tags for nth word. The tag of previous word which was selected 
# for the calculation for a tag of the current word is saved in v_max_tag. This
# is then used to retrieve the tags for all the words after calculation
# of the probability for the last word.
# posterior(): The logarithmic posterior is calculated for each algorithm
# The accuracies are as below:
# 
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       91.83%               38.30%
#         2. HMM VE:       92.02%               38.70%
#        3. HMM MAP:       93.16%               44.40%
#----
#
#External reference : #https://www.accelebrate.com/blog/using-defaultdict-python/
####


import random
import math
from collections import defaultdict


default_prb=0.00001
class Solver:
    
    
    # Emission probabilities
    emsn_prb = defaultdict(lambda: default_prb)
    # Initial probabilities
    init_prb = defaultdict(lambda: default_prb)
    # Transition probabilities
    trns_prb = defaultdict(lambda: default_prb)
    # Probability of a part of Speech tag
    pos_tag_prb = {}
    # Probability of a tag given word
    pos_tag_given_word_prb = defaultdict(lambda: default_prb)

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        first_word = True
        for word, pos_tag in zip(sentence, label):
            if first_word:
                posterior_v = math.log(self.pos_tag_prb[pos_tag] * self.emsn_prb[(word, pos_tag)], 10)
                first_word = False
            else:
                posterior_v += math.log(self.trns_prb[(pos_tag, prev_pos_tag)] * self.emsn_prb[(word, pos_tag)], 10)
                
                break
            prev_pos_tag = pos_tag 
        return posterior_v 
        

    # Do the training!
    
    def train(self, data):
        emsn_count = defaultdict(int)
        initial_count  = defaultdict(int)
        trns_count = defaultdict(int)
        word_count = defaultdict(int)
        pos_tag_count = defaultdict(int)
        
        for sentence in data:
            first_word = True
            for (word, pos_tag) in zip(sentence[0], sentence[1]):
                if first_word:
                    initial_count[pos_tag] += 1
                    first_word = False
                else :
                    trns_count[(pos_tag, prev_pos_tag)] += 1
                    

                emsn_count[(word, pos_tag)] += 1
                word_count[word] += 1
                pos_tag_count[pos_tag]   += 1  
                prev_pos_tag = pos_tag

        no_of_sentences = len(data)
        no_of_words = float(sum(word_count.values()))

        # Calculating Initial probabilities
        for i in initial_count:
            self.init_prb[i] = float(initial_count[i])/no_of_sentences

        # Calculating Transition Probabilities
        for i in trns_count:
            pos_tag, prev__pos_tag = i
            self.trns_prb[i] = float(trns_count[i])/pos_tag_count[prev__pos_tag]
        # Calculating POS Tag probabilities
        for i in pos_tag_count:
            self.pos_tag_prb[i] = pos_tag_count[i]/no_of_words

        # Calculating emission probabilities and P(S|W) using Bayes Law
        for i in emsn_count:
            word, pos_tag = i
            self.emsn_prb[i] = float(emsn_count[i])/pos_tag_count[pos_tag]
            # Using Bayes Law
            self.pos_tag_given_word_prb[(pos_tag, word)] = self.emsn_prb[i] * self.pos_tag_prb[pos_tag] / ( word_count[word]/no_of_words )

    # Algorithm Functions

    def simplified(self, sentence):
        pos_tags = [ "noun" ] * len(sentence)
        probs = [0] * len(sentence)
        for i, word in enumerate(sentence):
            for pos_tag in self.pos_tag_prb:
                if probs[i] < self.pos_tag_given_word_prb[(pos_tag, word)]:
                    pos_tags[i] = pos_tag
                    probs[i] = self.pos_tag_given_word_prb[(pos_tag, word)] 

        return ( pos_tags )
        
		
    def hmm_ve(self, sentence):
        pos_tags = [ "noun" ] * len(sentence)
        probs = [0] * len(sentence)
        p = defaultdict(int)
        first_word = True
        for i, word in enumerate(sentence):
            for pos_tag in self.pos_tag_prb:
            #check if it's the start of the sentence
                if first_word:
                    for pos_tag_1 in self.pos_tag_prb:
                        p[(i,pos_tag_1)] = self.init_prb[pos_tag_1]*self.emsn_prb[(word,pos_tag_1)]
                        if(p[(i,pos_tag_1)] > probs[i] ) :
                            probs[i] = p[(i, pos_tag_1)]
                            pos_tags[i] = pos_tag_1

                    first_word = False
                else:
                    for prev_pos_tag in self.pos_tag_prb:
                        p[(i,pos_tag)] += self.trns_prb[(pos_tag, prev_pos_tag)] * p[(i-1, prev_pos_tag)] 
                    
                p[(i,pos_tag)] *= self.emsn_prb[(word,pos_tag)] 
		
        for i, word in enumerate(sentence):
            for pos_tag in self.pos_tag_prb:
                if probs[i] < p[(i, pos_tag)]:
                    probs[i] = p[(i, pos_tag)]
                    pos_tags[i] = pos_tag
            
        return ( pos_tags )
		
    def hmm_viterbi(self, sentence):
        pos_tags = [ "noun" ] * len(sentence)
        probs = [0] * len(sentence)

        v = defaultdict(int)
        # Holds the tag of the max prev v(tag) selected for current v(tag)
        v_max_pos_tag = {}

        first_word = True
        for i, word in enumerate(sentence):
            for pos_tag in self.pos_tag_prb:
                if first_word:
                    v[(i,pos_tag)] = self.init_prb[pos_tag]
                else:
                    for prev_pos_tag in self.pos_tag_prb:
                        if v[(i, pos_tag)] < v[(i-1, prev_pos_tag)] * self.trns_prb[(pos_tag, prev_pos_tag)]:
                            v[(i, pos_tag)] = v[(i-1, prev_pos_tag)] * self.trns_prb[(pos_tag, prev_pos_tag)]
                            v_max_pos_tag[(i, pos_tag)] = prev_pos_tag

                v[(i,pos_tag)] *= self.emsn_prb[(word,pos_tag)]
                    
            first_word = False

	# Select the tag for last word with max probability
        max_last_prb = 0.0
        for pos_tag in self.pos_tag_prb:
            if max_last_prb < v[(i,pos_tag)]:
                max_last_prb = v[(i,pos_tag)]
                pos_tags[i] = pos_tag

        probs[i] = v[(i,pos_tags[i])]

	# Trace back the tags for all the words
        i -= 1
        while i >= 0:
            pos_tags[i] = v_max_pos_tag[(i + 1, pos_tags[i + 1])]
            probs[i] = v[(i,pos_tags[i])]
            i -= 1
         
        return ( pos_tags )

    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print ("Unknown algo!")
