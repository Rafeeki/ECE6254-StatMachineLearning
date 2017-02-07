######################
## Ryan Gentry
## ECE 6254 HW #1
## Jan 28, 2016
######################

import math

### Problem 1: Probability P of k heads from n flips of a single
### coin with probability p of heads in a single toss
### P[k|n, p] = ((n!/((n-k)!k!)*(p^k)*(1-p)^(n-k)

## Problem 1a: Compute exact probability of at least one of m
## coins with k = 0, given p & m.

# initialize variables
n = 10
p = [0.05, 0.75]
m = [1, 1000, 1000000]

# outer loop for each probability p
for i in range(0,len(p)):
	#P_0 = Prob a coin has k = 0
	P_0 = (1-p[i])**n
	print("If the probability of heads =  " + str(p[i]) + ", then P[0|10,0.05] = " + str(P_0)  + "...")

# inner loop for each # of coins m
	for j in range(0,len(m)):
		#P_1 = Prob a coin does not have k = 0
		P_1 = 1 - P_0
		#P_n = Prob 0 out of m coins has k = 0
		P_n = P_1**(m[j])
		#P_f = Prob at least 1 coin out of m has k = 0
		P_f = round(1-P_n,4)
		print("For " + str(m[j]) + " coins, probability of at least 1 coins with 0 heads = " + str(P_f))

