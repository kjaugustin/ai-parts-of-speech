# Parts of speech tagging using HMM

We use Bayesian network to model this problem. A Bayesnet with a set of N random
variables S = {S1; : : : ; SN} representing the part of speech tags and N random variables 
W = {W1; : : : ;WN} represnting the words.

Steps:

1. Estimate the probabilities of the HMM

2. Label new sentences with parts of speech, using the probability distributions learned
in step 1.

3. Implement a better model that incorporates dependencies between words. Implement Variable Elimination.

4. Finally, implement the Viterbi algorithm to find the maximum a posteriori (MAP) labeling for the
sentence
