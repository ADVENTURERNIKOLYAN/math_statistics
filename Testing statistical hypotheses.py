import numpy as np
import random
from collections import Counter
from scipy.special import betainc
from math import floor, log, e
import matplotlib as plot
import scipy
import statsmodels.api
from statsmodels.distributions.empirical_distribution import ECDF

'''
CHI-square (Pearson) goodness-of-fit test
'''

n = 228
p = 0.228
q = 1 - p

def Binomka(p, n):
    c = p/(1-p)
    s = (1 - p) ** n
    r = s
    k = 0
    a = random.uniform(0, 1)
    while a > s:
        k += 1
        r = r * c * (n - k + 1)/k
        s = s + r
    return k

def sample(size):
    sample = []
    for a in range(size):
        sample.append(int(Binomka(p, n)))
    return sample

sample = sample(1000)
sample.sort()

#fix the sample

sample = [34, 35, 36, 36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 69, 69, 70, 70, 70, 70, 73]

frequencyVector = list(dict(Counter(sample)).values())

'''
Vector of theoretical frequencies

Since the criterion ????2 err on samples with low-frequency events, so we take the interval [35,68]

Find ???????? as the difference between two distribution functions (end - beginning) ??? ????????=????(????????)???????(????????) - this will be the probability of the observation falling into the j-th interval when the hypothesis is fulfilled ????0

Distribution function ????1 - ????(?????????????????,1+??????????) - this is an incomplete Betta function with parameters  (?????????????????,1+??????????,1???????)
'''

teoreticalVector = []
for i in range(35,69):
    teoreticalVector.append(((betainc(n-floor(i+1),1+floor(i+1),1-p)) - betainc(n-floor(i),1+floor(i),1-p))*1000)
print(teoreticalVector)

'''
Let's calculate the measure of the difference  ????2 
"Cut off" unlikely events
'''

frequencyVector = frequencyVector[2:-2:]

'''
Checking that there is no dimension error
'''

print(len(frequencyVector),len(teoreticalVector))

'''
Now let's calculate the criterion  ????2  through the formula:  ????2=???????=1????(???????????????????????)2????????????
'''

Chi = 0
for i in range(len(teoreticalVector)):
    Chi += ((frequencyVector[i]-teoreticalVector[i])**2)/(teoreticalVector[i])
print(Chi)

'''
The Pearson chi-square goodness test is formulated as follows:  ????0???????2<???????? ,  ????????=????21 - ????(????- 1 ) , where N is the number of frequencies.
We have a division into 34 non-intersecting intervals.
Hence the degree of freedom of distribution ????2 = 34 - 1 = 33
Degree of freedom ???? = 0.05
Therefore, the table ????2( 33 )  = 47.3999
????2 (33)> 46.399024413207215
In our case, the hypothesis ????0  not rejected
'''

'''
CHI-square (Pearson) goodness-of-fit test for complex hypothesis
'''

k = 228
p = 0.228
q = 1 - p

def Binomka(p, n):
    c = p/(1-p)
    s = (1 - p) ** n
    r = s
    k = 0
    a = random.uniform(0, 1)
    while a > s:
        k += 1
        r = r * c * (n - k + 1)/k
        s = s + r
    return k

def sample(size):
    sample = []
    for a in range(size):
        sample.append(int(Binomka(p, k)))
    return sample

sampleComplexBinominal = sample(1000)
sampleComplexBinominal.sort()
'''
We fix the sample
'''
sampleComplexBinominal=[35, 35, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 68, 68, 68, 69, 70, 70, 71 , 71 , 71 , 72 ]
'''
Next, you need to estimate the unknown parameter ??.
To do this, use the maximum likelihood estimate for a vector with a polynomial distribution: ?????? = ???? ???? ???? ???? ???????????????????=1????(???????????( ???? ))????????
From the 3rd long-term we have: ?????? =????(????)=one??????????? = 1????????????????, at ???? = 228
'''

X_i_Bi = 0
for i in range(len(sampleComplexBinominal)):
    X_i_Bi += abs(sampleComplexBinominal[i])
X_i_Bi = X_i_Bi / 1000
ThetaWhitAHatBi =X_i_Bi/k
print(ThetaWhitAHatBi)

'''
Our point estimate  ???  to our given estimate, which means we did everything right

Let's calculate the frequency of outcomes by the formula ????????=??????? = 1????????(????????= ???? ) , ???? = 1 , . . . , ????
'''

frequencyComplexBinominalVector=list(dict(Counter(sampleComplexBinominal)).values());
print(dict(Counter(sampleComplexBinominal)))

'''
Find  ????????????  as the difference between two distribution functions (end - beginning)  ???   ???????????? =????(????????, ???? ) - ????(????????, ???? ) - this will be the probability of the observation falling into the j-th interval
Distribution function for the binomial distribution: ????1 - ????(?????????????????,1+??????????)  - this is an incomplete Betta function with parameters  (?????????????????,1+??????????,1???????) 
Since the criterion ????2 err on samples with low-frequency events, so we take the interval [39.68]

'''

teoreticalComplexBinominalVector  = []
for  i  in  range ( 39 , 69 ):
    teoreticalComplexBinominalVector.append(((betainc(k-floor(i+1),1+floor(i+1),1.0-ThetaWhitAHatBi)) - betainc(k-floor(i),1+floor(i),1.0-ThetaWhitAHatBi)))
print ( teoreticalComplexBinominalVector )
'''
Let us check that the dimensions of the frequency and theoretical vectors coincide
'''

frequencyComplexBinominalVector = frequencyComplexBinominalVector[4:-4:]
print(len(frequencyComplexBinominalVector),len(teoreticalComplexBinominalVector))

ChiComplexBi = 0
for i in range(len(teoreticalComplexBinominalVector)):
    ChiComplexBi += ((frequencyComplexBinominalVector[i]-1000*teoreticalComplexBinominalVector[i])**2)/(1000*teoreticalComplexBinominalVector[i])
print(ChiComplexBi)

'''
The Pearson chi-square goodness test is formulated as follows:  ????0???????2<???????? ,  ????????=????21 - ????(???????1???????) , where N is the number of frequencies, r is the rank of the matrix || ????????????( ???? ) / ???? (????????) ||
We have a partition into 30 non-intersecting intervals.
Hence the degree of freedom of distribution ????2 = 30 - 1 - 1 = 28
Degree of freedom ???? = 0.05
Therefore, the table ????2( 28 )  = 41.3371
????2 (28)> 40.95682855025718
In our case, the hypothesis ????0  not rejecteds
'''

'''
ChI-square (Pearson) uniformity test
'''

'''
Let there be m series of independent observations consisting of  ????one, . ... ... ,????????  observations, respectively, and let  ????????= (???????? 1, . ... ... ,????????????)  - the frequency of outcomes of the i-series, and  ????????= (???????? 1, . ... ... ,????????????) - their probabilities (i = 1, ..., m).
Then the hypothesis of homogeneity means the statement that the probabilities of outcomes did not change from series to series, i.e.
????0:????one= . ... ... =????????= ???? = (????one, . ... ... ,????????) 
Where p is some unknown vector of probabilities  ????one+ . ... ... +????????= 1 
As in the case of using the chi-square test for the case of a complex hypothesis, we replace the values ???????? their maximum likelihood estimate ?????? ????built for all samples:
?????? = (?????? one, . ... ... ,?????? ????) = ???? ???? ???? ???? ???????????????????,????????????????????,????????= ???? ???? ???? ???? ???????????????????????????????????,????????
'''

k = 228
p = 0.228
q = 1 - p

def Binomka(p, n):
    c = p/(1-p)
    s = (1 - p) ** n
    r = s
    k = 0
    a = random.uniform(0, 1)
    while a > s:
        k += 1
        r = r * c * (n - k + 1)/k
        s = s + r
    return k

def sample(size):
    sample = []
    for a in range(size):
        sample.append(int(Binomka(p, k)))
    return sample

sampleHomogeneousBinominal = sample(1000)
sampleHomogeneousBinominal.sort()

'''
Fixing 3 random samples
'''

sampleHomogeneousBinominal_1 = [35, 36, 36, 37, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 68, 69, 69, 69, 69, 70, 71, 72]
sampleHomogeneousBinominal_2 = [35, 35, 35, 35, 36, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 67, 67, 68, 68, 69, 69, 70, 71, 72]
sampleHomogeneousBinominal_3 = [35, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 67, 67, 67, 67, 68, 68, 68, 69, 69, 69, 69 , 70 , 70 , 71 , 72 ]

'''
Looking for the frequency in each sample
'''

frequencyHomogeneousBinominalVector_1 = list(dict(Counter(sampleHomogeneousBinominal_1)).values())
frequencyHomogeneousBinominalVector_2 = list(dict(Counter(sampleHomogeneousBinominal_2)).values())
frequencyHomogeneousBinominalVector_3 = list(dict(Counter(sampleHomogeneousBinominal_3)).values())

frequencyHomogeneousBinominalVector123 = [frequencyHomogeneousBinominalVector_1,frequencyHomogeneousBinominalVector_2,frequencyHomogeneousBinominalVector_3]
threeSamplesBinominal = [sampleHomogeneousBinominal_1,sampleHomogeneousBinominal_2,sampleHomogeneousBinominal_3]

'''
We have 3 samples (m = 3), N = 228, ????one=????2=????3= 1000 ??? ???? =????one+????2+????3= 3000
The samples contain numbers from 35 to 72, let's check the probability of these numbers
'''

P  = [ 0 ] * 38
N = 3000
frequencyHomogeneousBinominalVector =[0]*38

for  i  in  range ( 3 ):
    for  j  in  range ( 1000 ):
        for t in range(35,73):
            if threeSamplesBinominal[i][j] == t:
                frequencyHomogeneousBinominalVector[t-35] = frequencyHomogeneousBinominalVector[t-35] + 1

for  i  in  range ( 38 ):
    P[i] = frequencyHomogeneousBinominalVector[i]/N

'''
Now we calculate the implementation of statistics ?????? ????2=(????eleven-????one?????? one)2????one?????? one+(????12-????one?????? 2)2????one?????? 2+ . ... ... +(????338-????3?????? 38)2????3?????? 38
'''

TWithAHatBinominal = 0
for  i  in  range ( 3 ):
    for  j  in  range ( 38 ):
        TWithAHatBinominal += ((frequencyHomogeneousBinominalVector123[i][j] - 1000*P[i])**2)/1000*P[i]
print(TWithAHatBinominal)
'''
The Pearson chi-square homogeneity criterion is formulated as follows: ????0???????2<????????, ????????=????21 - ????((???????1)(????- 1 ) ), where N is the number of frequencies, k is the number of samples
Now let's compare the obtained value with the boundary????21 - ????((???????1)(????- 1 ) ) =????20.95( 74 ) = 95.0015
0.8947334191851852 << 95.0015 ??? the data do not contradict the hypothesis ??? hypothesis ????0 not rejected
'''

'''
Laplace distribution
'''

'''
Take the segment [-2: 2] ( ????= 0 ), on which we will construct our sample according to the Laplace distribution, since on the interval  ( - ??? ; 2 ) ??? ( 2 ; ??? )
'''

'''
Let's set the parameters
'''

n = 1488
alpha = 0.228
betta = 0.0

def LaplaceDistribution(alpha):
    e1 = -(log(random.uniform(0, 1)) / alpha)
    e2 = -(log(random.uniform(0, 1)) / alpha)
    return e1 - e2

def sample(size):
    sample = []
    for a in range(size):
        sample.append(LaplaceDistribution(alpha))
    return sample

sampleLaplace = sample(10000)
sampleLaplace.sort()

'''
Since we take the interval [-2; 2], we will remove from the selection all values ??????less than -2 and more than 2
'''

timeSample = []
for i in range(10000):
    if (round(sampleLaplace[i], 2) <= 2) and (-2<=round(sampleLaplace[i], 2)):
        timeSample.append(round(sampleLaplace[i], 1))
sampleLaplace = timeSample

frequencyVectorLaplace = list(dict(Counter(sampleLaplace)).values())
print(frequencyVectorLaplace)

def distrubution_func(x):
    if x<=int(betta):
        return (1/2)*(e)**(alpha*(x-int(betta)))
    if x>int(betta):
        return 1-(1/2)*(e)**(-alpha*(x-int(betta)))

'''
Let us construct a vector of theoretical frequencies. Find ????????  as the difference between two distribution functions (end - beginning)  ???   ????????=????(????????)???????(????????)  - this will be the probability of the observation falling into the j-th interval when the hypothesis is fulfilled  ????0 
Distribution function for the Laplace distribution:
{one/2^exp^(???? ( ???? - ????)); 1 - one/2^exp^(- ???? ( ???? - ????)), ???? ??? ????,????>????
'''

teoreticalVectorLaplace = []
sampleLaplace = list(set(sampleLaplace))
sampleLaplace.sort()
for i in range(len(sampleLaplace)-1):
    teoreticalVectorLaplace.append((distrubution_func(sampleLaplace[i+1]) - distrubution_func(sampleLaplace[i]))*10000)
teoreticalVectorLaplace = list(teoreticalVectorLaplace)

'''
Checking the dimension of the frequency and theoretical vectors
'''

frequencyVectorLaplace = frequencyVectorLaplace[1::]
print(len(frequencyVectorLaplace),len(teoreticalVectorLaplace))

ChiLaplace = 0
number = 10000
for i in range(len(teoreticalVectorLaplace)):
    ChiLaplace += ((frequencyVectorLaplace[i]-teoreticalVectorLaplace[i])**2)/(teoreticalVectorLaplace[i])
print(ChiLaplace)

'''
The Pearson chi-square goodness test is formulated as follows:  ????0???????2<???????? ,  ????????=????21 - ????(????- 1 ) , where N is the number of frequencies.
We have a partition into 40 non-intersecting intervals.
Hence the degree of freedom of distribution ????2 = 40 - 1 = 39
Degree of freedom ???? = 0.05
Therefore, the table ????2( 39 )  = 54.5722
????2( 39 ) > 31.746211664074156
So we do not reject the hypothesis ????0
'''

'''
CHI-square (Pearson) goodness-of-fit test for complex hypothesis
'''

'''
Let's set the parameters
'''

n = 1488
alpha = 0.228
betta = 0.0

def LaplaceDistribution(alpha):
    e1 = -(log(random.uniform(0, 1)) / alpha)
    e2 = -(log(random.uniform(0, 1)) / alpha)
    return e1 - e2

def sample(size):
    sample = []
    for a in range(size):
        sample.append(LaplaceDistribution(alpha))
    return sample

'''
We build a sample of size 10000
'''

sampleComplexLaplace = sample(10000)
sampleComplexLaplace.sort()

'''
Next, you need to estimate the unknown parameter ??.
To do this, use the maximum likelihood estimate from the 3rd long term: ?????? =????(????)=oneone??????????? = 1????|????????| , at  ????= 0
'''

X_i = 0
for i in range(len(sampleComplexLaplace)):
    X_i += abs(sampleComplexLaplace[i])
ThetaWhitAHat = 10000/X_i
print(ThetaWhitAHat)

'''
Our point estimate  ???  to our given estimate, which means we did everything right

Let's select the values ??????from -2 to 2, since in this interval the probability is
maximal.Calculate the frequencies of outcomes using the formula????????=??????? = 1????????(????????= ???? ) , ???? = 1 , . . . , ????
'''

x = []
for i in range(10000):
    if (round(sampleComplexLaplace[i], 2) <= 2) and (-2<=round(sampleComplexLaplace[i], 2)):
        x.append(round(sampleComplexLaplace[i], 1))
sampleComplexLaplace = x

dict(Counter(sampleComplexLaplace));
frequencyVectorComplexLaplace = list(dict(Counter(sampleComplexLaplace)).values())
frequencyVectorComplexLaplace = frequencyVectorComplexLaplace[1::]
print(list(dict(Counter(sampleComplexLaplace)).values()))

'''
Now we find the theoretical frequency depending on  ???? , by the formula: 
Find ????????  as the difference between two distribution functions (end - beginning)  ???   ????????=????(????????)???????(????????)  - this will be the probability of the observation falling into the j-th interval when the hypothesis is fulfilled  ????0 
Distribution function for the Laplace distribution:
{one/2^exp^(???? ( ???? - ????)); 1 - one/2^exp^(- ???? ( ???? - ????)), ???? ??? ????,????>????
'''

complexTeoreticalVectorLaplace =[]
sampleComplexLaplace = list(set(sampleComplexLaplace))
sampleComplexLaplace.sort()

def distrubution_func(x,alpha):
    if x<=int(betta):
        return (1/2)*(e)**(alpha*(x-int(betta)))
    if x>int(betta):
        return 1-(1/2)*(e)**(-alpha*(x-int(betta)))

for i in range(len(sampleComplexLaplace)-1):
    p = distrubution_func(sampleComplexLaplace[i+1],ThetaWhitAHat) - distrubution_func(sampleComplexLaplace[i],ThetaWhitAHat)
    complexTeoreticalVectorLaplace.append(p*10000)

print(len(frequencyVectorComplexLaplace),len(complexTeoreticalVectorLaplace))

'''
To construct a criterion, we use the formula:
??????2????=????2????(??????????) =???????=1????(???????????????????????(??????????))2????????????(??????????) 
'''

ChiComplexLaplace = 0
for i in range(len(complexTeoreticalVectorLaplace)):
    ChiComplexLaplace += ((frequencyVectorComplexLaplace[i]-complexTeoreticalVectorLaplace[i])**2)/(complexTeoreticalVectorLaplace[i])
print(ChiComplexLaplace)

'''
The Pearson chi-square goodness test is formulated as follows:  ????0???????2<???????? ,  ????????=????21 - ????(???????1???????) , where N is the number of frequencies, r is the rank of the matrix || ????????????( ???? ) / ???? (????????) ||
We have a partition into 30 non-intersecting intervals.
Hence the degree of freedom of distribution ????2 = 40 - 1 - 1 = 38
Degree of freedom ???? = 0.05
Therefore, the table ????2( 38 )  = 53.3835
????2 (38)> 49.329478561056966
In our case, the hypothesis ????0  not rejected
'''

'''
ChI-square (Pearson) uniformity test
'''

'''
Let there be m series of independent observations consisting of  ????one, . ... ... ,????????  observations, respectively, and let  ????????= (???????? 1, . ... ... ,????????????)  - the frequency of outcomes of the i-series, and  ????????= (???????? 1, . ... ... ,????????????) - their probabilities (i = 1, ..., m).
Then the hypothesis of homogeneity means the statement that the probabilities of outcomes did not change from series to series, i.e.
????0:????one= . ... ... =????????= ???? = (????one, . ... ... ,????????) 
Where p is some unknown vector of probabilities  ????one+ . ... ... +????????= 1 
As in the case of using the chi-square test for the case of a complex hypothesis, we replace the values ???????? their maximum likelihood estimate ?????? ????built for all samples:
?????? = (?????? one, . ... ... ,?????? ????) = ???? ???? ???? ???? ???????????????????,????????????????????,????????= ???? ???? ???? ???? ???????????????????????????????????,????????
'''

n = 1488
alpha = 0.228
betta = 0.0

def LaplaceDistribution(alpha):
    e1 = -(log(random.uniform(0, 1)) / alpha)
    e2 = -(log(random.uniform(0, 1)) / alpha)
    return e1 - e2

def sample(size):
    sample = []
    for a in range(size):
        sample.append(LaplaceDistribution(alpha))
    return sample

'''
Create 3 selections, sort them, remove values that are less than -2 and more than 2
'''

sampleHomogeneousLaplace_1 = []
sampleHomogeneousLaplace_2 = []
sampleHomogeneousLaplace_3 = []
sampleHomogeneousLaplace123 = [sampleHomogeneousLaplace_1, sampleHomogeneousLaplace_2, sampleHomogeneousLaplace_3]
for j in range(3):
    sampleHomogeneousLaplace123[j] = sample(1000)
    sampleHomogeneousLaplace123[j].sort()
    timeSample = []
    for i in range(1000):
        if (round(sampleHomogeneousLaplace123[j][i], 2) <= 2) and (-2<=round(sampleHomogeneousLaplace123[j][i], 2)):
            timeSample.append(round(sampleHomogeneousLaplace123[j][i], 1))
    sampleHomogeneousLaplace123[j] = timeSample
sampleHomogeneousLaplace = [list(dict(Counter(sampleHomogeneousLaplace123[0])).values()),list(dict(Counter(sampleHomogeneousLaplace123[1])).values()),list(dict(Counter(sampleHomogeneousLaplace123[2])).values())]

'''
First, we find the sum of all repeating elements in three samples, and then we look for the probability of their dropping
out.We have 3 samples (m = 3), N = 228, ????one=????2=????3= 1000 ??? ???? =????one+????2+????3= 3000 
'''

P = [0]*len(set(sampleHomogeneousLaplace123[0]))
N = 3000
frequencyHomogeneousLaplaceVector =[0]*len(set(sampleHomogeneousLaplace123[0]))
Mirror = list((set(sampleHomogeneousLaplace123[0])))
Mirror.sort()

for i in range(3):
    for j in range(len(sampleHomogeneousLaplace[i])):
        for t in range(len(Mirror)):
            if sampleHomogeneousLaplace123[i][j] == Mirror[t]:
                frequencyHomogeneousLaplaceVector[t] = frequencyHomogeneousLaplaceVector[t] + 1

for i in range(len(Mirror)):
    P[i] = frequencyHomogeneousLaplaceVector[i]/N

'''
Now we calculate the implementation of statistics using the formula:  ?????? ????2=(????11???????1?????? 1)2????1?????? 1+(????12???????1?????? 2)2????1?????? 2+...+(????341???????3?????? 41)2????3?????? 41
'''

TWithAHatLaplace = 0
for i in range(3):
    for j in range(len(Mirror)):
        TWithAHatLaplace += ((sampleHomogeneousLaplace[i][j] - 1000*P[i])**2)/1000*P[i]
print(TWithAHatLaplace)

print('N =',len(Mirror))

'''
The Pearson chi-square homogeneity criterion is formulated as follows:  ????0???????2<???????? ,  ????????=????21 - ????((???????1)(????- 1 ) ) , where N is the number of frequencies, k is the number of samples
Now let's compare the obtained value with the boundary ????21 - ????((???????1)(????- 1 ) ) =????20.95( 80 ) = 101.879 
?????? ????2<< 101.879 ??? the data do not contradict the hypothesis ??? hypothesis ????0 not rejected
'''

'''
Kolmogorov-Smirnov goodness test
'''

'''
The Kolmagorov-Smornov criteria are applied when the function F (x) is continuous. 
The criterion statistics are determined by the formula ????????=????????(????) =???? ???? ???????inf<????<inf|?????? ????(????)???????(????)|  
and represents the maximum deviation of the empirical distribution function  ?????? ????(????) , 
built on the basis of the sample X, from the hypothetical (i.e., determined by the hypothesis  ????0 ) 
the distribution function F (x)
'''

n = 1488
alpha = 0.228
betta = 0.0

def LaplaceDistribution(alpha):
    e1 = -(log(random.uniform(0, 1)) / alpha)
    e2 = -(log(random.uniform(0, 1)) / alpha)
    return e1 - e2

def distrubution_func(x):
    if x<=int(betta):
        return (1/2)*(e)**(alpha*(x-int(betta)))
    if x>int(betta):
        return 1-(1/2)*(e)**(-alpha*(x-int(betta)))

def sample(size):
    sample = []
    for a in range(size):
        sample.append(LaplaceDistribution(alpha))
    return sample

sampleKolmagorovLaplace = sample(10000)
sampleKolmagorovLaplace.sort()

'''
We create an EGF function for our sample and write down the values
'''

ECDF=statsmodels.distributions.empirical_distribution.ECDF(sampleKolmagorovLaplace,side='right')
ECDF.y = ECDF.y[1::];

'''
We are looking for the maximum deviation
'''

Dn = []
for i in range(len(ECDF.y)):
    Dn.append(abs(ECDF.y[i]-distrubution_func(sampleKolmagorovLaplace[i])))

'''
The Kolmagorov-Smirnov goodness-of-fit criterion is formulated as follows: if  ???? ??? 20  and at the chosen level of significance  ????= 0.05  number  ????????  defined by the relation  ????(????????) = 1 - ???? where  ????????=????????/????????????? 
You need to check the condition ????(?????????????????????<?????????????????????) ??? ????(????????) = ??????? ????(?????????????????????????????????????????????) ??? ????(????????) = 1 - ????where ????????=???????????????
If a ????????<??????????? hypothesis ????0 accepted
'''

T_alpha = 1.36/(10000**(1/2))
print(max(Dn))
print(T_alpha)

'''
In our case  ????????<???????? , therefore, the hypothesis  ????0  accepted
'''

'''
Kolmogorov-Smirnov goodness test for a complex hypothesis
'''

'''
The Kolmagorov-Smornov criteria are applied when the function F (x) is continuous. 
The criterion statistics are determined by the formula ????????=????????(????) =???? ???? ???????inf<????<inf|?????? ????(????)???????( ???? , ???? ) |  
and represents the maximum deviation of the empirical distribution function  ?????? ????(????) , 
built on the basis of the sample X, from the hypothetical (i.e., determined by the hypothesis  ????0 ) 
the distribution function F (x)
'''

n = 1488
alpha = 0.228
betta = 0.0

def LaplaceDistribution(alpha):
    e1 = -(log(random.uniform(0, 1)) / alpha)
    e2 = -(log(random.uniform(0, 1)) / alpha)
    return e1 - e2

def sample(size):
    sample = []
    for a in range(size):
        sample.append(LaplaceDistribution(alpha))
    return sample

sampleComplexKolmagorovLaplace = sample(1000)
sampleComplexKolmagorovLaplace.sort()

'''
We create an EGF function for our sample and write down the values
'''

ECDFComplex=statsmodels.distributions.empirical_distribution.ECDF(sampleComplexKolmagorovLaplace,side='right')
ECDFComplex.y = ECDFComplex.y[1::]

'''
Next, you need to estimate the unknown parameter ??.
To do this, use the maximum likelihood estimate from the 3rd long term:?????? =????(????)=oneone??????????? = 1????|????????|, at ????= 0
'''

timeSample = 0
for i in range(len(sampleComplexKolmagorovLaplace)):
    timeSample += abs(sampleComplexKolmagorovLaplace[i])
ThetaWhitAHatComplexKolmagorov = 1000/timeSample
print(ThetaWhitAHatComplexKolmagorov)

'''
Our point estimate ??? to our given estimate, which means we did everything right
Distribution function for the Laplace distribution:
{one2exp???? ( ???? - ????)1 -one2exp- ???? ( ???? - ????), ???? ??? ????,????>????
'''

def distrubution_func(x,alpha):
    if x<=int(betta):
        return (1/2)*(e)**(alpha*(x-int(betta)))
    if x>int(betta):
        return 1-(1/2)*(e)**(-alpha*(x-int(betta)))
CDF = []
for  i  in  range ( 1000 ):
    CDF.append(distrubution_func(sampleComplexKolmagorovLaplace[i],ThetaWhitAHatComplexKolmagorov))
'''
The Kolmagorov-Smirnov goodness-of-fit criterion is formulated as follows: if ???? ??? 20 and at the chosen level of significance ????= 0.05 number ???????? defined by the relation ????(????????) = 1 - ????where ????????=????????/?????????????
You need to check the condition ????(?????????????????????<?????????????????????) ??? ????(????????) = ??????? ????(?????????????????????????????????????????????) ??? ????(????????) = 1 - ????where ????????=???????????????
If a ????????<??????????? hypothesis ????0 accepted

We are looking for the maximum deviation
'''

DnComplex = []
for  i  in  range ( len ( ECDFComplex . y )):
    DnComplex.append(abs(ECDFComplex.y[i]-CDF[i]))
T_alphaComplex = 1.36/(1000**(1/2))
print(max(DnComplex))
print(T_alphaComplex)

'''
In our case ????????<????????, therefore, the hypothesis ????0 accepted
'''

'''
Kolmogorov-Smirnov homogeneity criterion 
'''

'''
This criterion is needed in order to compare 2 different samples and understand whether they belong to the same distribution or not.
????????,????=???? ???? ???????inf<????<inf|?????? 1 ????(????)????????? 1 ????(????)| where  ?????? 1 ????(????),?????? 1 ????(????)  - empirical distribution functions based on different sampl
'''

n = 1488
alpha = 0.228
betta = 0.0

def LaplaceDistribution(alpha):
    e1 = -(log(random.uniform(0, 1)) / alpha)
    e2 = -(log(random.uniform(0, 1)) / alpha)
    return e1 - e2

def distrubution_func(x):
    if x<=int(betta):
        return (1/2)*(e)**(alpha*(x-int(betta)))
    if x>int(betta):
        return 1-(1/2)*(e)**(-alpha*(x-int(betta)))

def sample(size):
    sample = []
    for a in range(size):
        sample.append(LaplaceDistribution(alpha))
    return sample

'''
Making 2 selections
'''

sampleHomogeneousSmirnovLaplace = []
sampleHomogeneousSmirnovLaplace_1 = []
sampleHomogeneousSmirnovLaplace_2 = []
sampleHomogeneousSmirnovLaplace123 = [sampleHomogeneousSmirnovLaplace_1, sampleHomogeneousSmirnovLaplace_2]
for j in range(2):
    sampleHomogeneousSmirnovLaplace123[j] = sample(1000)
    sampleHomogeneousSmirnovLaplace123[j].sort()
    sampleHomogeneousSmirnovLaplace.append(sampleHomogeneousSmirnovLaplace123[j].sort())

def near_value(it, value):
    return min(it,key=lambda x:abs(x-value))

def near_func(sample,num):
    sample.sort()
    a=0
    c1=dict(sorted(Counter(sample).items()))
    temp=near_value(c1.keys(),num)
    if temp>=num:
        for j in range(0,list(c1.keys()).index(temp)):
            a+=c1[list(c1.keys())[j]]/len(sample)
    else:
        for j in range(0,list(c1.keys()).index(temp)+1):
            a+=c1[list(c1.keys())[j]]/len(sample)
    return a

'''
Looking for the maximum difference between two empirical functions
'''

DP=0
DM=0
N = 1000
for i in range(N):
    if DP<((i+1)/N)-near_func(sampleHomogeneousSmirnovLaplace[0],sampleHomogeneousSmirnovLaplace[1][i]):
        DP = ((i + 1) / N) - near_func(sampleHomogeneousSmirnovLaplace[0],sampleHomogeneousSmirnovLaplace[1][i])

    if DM<(near_func(sampleHomogeneousSmirnovLaplace[0],sampleHomogeneousSmirnovLaplace[1][i]) - i / N):
        DM = (near_func(sampleHomogeneousSmirnovLaplace[0],sampleHomogeneousSmirnovLaplace[1][i]) - i / N)

'''
Smirnov's criterion is: ????0of tons in e p r a e t a i ???????????,????>????????one????+one????????????????????????????????????????, where n, m are sample sizes
????0.95= 0.52
'''

Lambda_alpha = 0.52/(1/1000+1/1000)**(1/2)
print(max(DP,DM))
print(Lambda_alpha)

'''
In our case ????0of tons in e p r a e t a i ???????????,????<????????one????+one???????????????????????????????????????? ??? criterion ????0 not rejected
'''
