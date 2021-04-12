import random
from math import log, e, floor
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.special import betainc

n = 1488
alfa = 0.228
bettaRandom = (abs(random.uniform(0, 2)))
ar = (abs(random.uniform(0, 2)))
betta = 0.0
sampling = [[], [], [], [], []]
znacheniya_Y = []
znacheniaOfBinominalFunction = []
viborkaIsFive = [[], [], [], [],  []]
znacheniaViborkiIsFive = [[], [], [], [], []]


def LaplaceDistribution(alfa):
    e1 = -(log(random.uniform(0, 1)) / alfa)
    e2 = -(log(random.uniform(0, 1)) / alfa)
    return e1 - e2


def modelTheoreticalLaplaceDistribution(alfa):
    Os_x = []
    Os_y = []
    for a in range(-20, 20):
        Os_x.append(a)
    for a in Os_x:
        Os_y.append(e**(-alfa*abs(a - betta))*alfa/2)
    plt.plot(Os_x, Os_y, label='Teoretical', color='black')
    plt.show()


def randomLaplace(x, alfa, betta):
    return (alfa / 2) * e ** (-alfa * abs(x - betta))


def viborka(size):
    for j in range(5):
        for a in range(size):
            viborkaIsFive[j].append(LaplaceDistribution(alfa))
    return viborkaIsFive


def modelGraphOfLaplace(n):
    counter = 0
    Os_of_x = []
    Os_of_y = []
    while counter <= n:
        Os_of_x.append(LaplaceDistribution(alfa))
        counter += 1
    Os_of_x.sort()
    for a in Os_of_x:
        Os_of_y.append((e ** (-alfa * abs(a - betta))) * alfa / 2)
    plt.plot(Os_of_x, Os_of_y)
    plt.show()


def GraphEmpir(size):
    ALLViborka = viborka(size)
    MaxValue = 0
    znacheniaViborkiIsFive = [[], [], [], [], []]
    FRX = []
    FRY = []
    for numberOfIndex in range(5):
        itemOfViborkiIs5 = ALLViborka[numberOfIndex]
        itemOfViborkiIs5.sort()

        for i in range(len(itemOfViborkiIs5)):
            if itemOfViborkiIs5[i] > MaxValue:
                MaxValue = itemOfViborkiIs5[i]
            if itemOfViborkiIs5[i] <= itemOfViborkiIs5[0]:
                znacheniaViborkiIsFive[numberOfIndex].append(0.0)
            elif (itemOfViborkiIs5[i] > itemOfViborkiIs5[0]) and (itemOfViborkiIs5[i] < itemOfViborkiIs5[len(itemOfViborkiIs5) - 1]):
                znacheniaViborkiIsFive[numberOfIndex].append(i / len(itemOfViborkiIs5))
            else:
                znacheniaViborkiIsFive[numberOfIndex].append(1.0)

    plt.step(ALLViborka[0], znacheniaViborkiIsFive[0], color='red', label='ECDF' + str(0))
    plt.step(ALLViborka[1], znacheniaViborkiIsFive[1], color='orange', label='ECDF' + str(1))
    plt.step(ALLViborka[2], znacheniaViborkiIsFive[2], color='yellow', label='ECDF' + str(2))
    plt.step(ALLViborka[3], znacheniaViborkiIsFive[3], color='green', label='ECDF' + str(3))
    plt.step(ALLViborka[4], znacheniaViborkiIsFive[4], color='blue', label='ECDF' + str(4))


    counter = 0
    for i in range(-10, 10):
        FRX.append(i)
        counter += 1
    for j in range(counter):
        if FRX[j] <= betta:
            y = (1 / 2) * (e ** (alfa * (FRX[j] - betta)))
            FRY.append(y)
        else:
            y = 1.0-(1/2)*(e**(-alfa*(FRX[j]-betta)))
            FRY.append(y)

    plt.plot(FRX, FRY, color='black', label='CDF' + str(5))
    plt.legend()
    plt.show()


def Quantile(size):
    Quantile = []
    sampling = viborka(size)
    sampling[0].sort()
    for v in [0.1, 0.5, 0.7]:
        k = floor(v * (size - 1))
        if k + 1 < v * size:
            Quantile.append(sampling[0][k + 1])
        elif k + 1 == v * size:
            Quantile.append((sampling[0][k] + sampling[0][k + 1]) / 2)
        elif k + 1 > v * size:
            Quantile.append(sampling[0][k])

    print(Quantile)


def frequency_polygon(size):
    Y_frequency_histogram = [0, 0, 0, 0, 0]
    sampling = viborka(size)
    sampling_frequantli = sampling.copy()

    for i in range(5):
        sampling_frequantli[i] = list(np.around(sampling_frequantli[i]))
        sampling_frequantli[i] = list(set(sampling_frequantli[i]))
        sampling[i] = list(np.around(sampling[i]))
        sampling[i].sort()
        sampling_frequantli[i].sort()
        Y_frequency_histogram[i] = [0] * len(sampling_frequantli[i])

    for i in range(5):
        for a in range(len(sampling_frequantli[i])):
            Y_frequency_histogram[i][a] = (sampling[i]).count(sampling_frequantli[i][a])/len(sampling[i])

    plt.step(sampling_frequantli[0], Y_frequency_histogram[0], label='EPMF' + str(1))
    plt.step(sampling_frequantli[1], Y_frequency_histogram[1], label='EPMF' + str(2))
    plt.step(sampling_frequantli[2], Y_frequency_histogram[2], label='EPMF' + str(3))
    plt.step(sampling_frequantli[3], Y_frequency_histogram[3], label='EPMF' + str(4))
    plt.step(sampling_frequantli[4], Y_frequency_histogram[4], label='EPMF' + str(5))

    Os_x = []
    Os_y = []
    for i in range(-10, 10):
        Os_x.append(i)
    for a in Os_x:
        Os_y.append((e ** (-alfa * abs(a - betta))) * alfa / 2)
    plt.plot(Os_x, Os_y, label='PMF')
    plt.legend()
    plt.show()


def frequency_histogram(size):
    Y_frequency_histogram = [0, 0, 0, 0, 0]
    sampling = viborka(size)
    sampling_frequantli = sampling.copy()

    for i in range(5):
        sampling_frequantli[i] = list(np.around(sampling_frequantli[i]))
        sampling_frequantli[i] = list(set(sampling_frequantli[i]))
        sampling[i] = list(np.around(sampling[i]))
        sampling[i].sort()
        sampling_frequantli[i].sort()
        Y_frequency_histogram[i] = [0] * len(sampling_frequantli[i])

    for i in range(5):
        for a in range(len(sampling_frequantli[i])):
            Y_frequency_histogram[i][a] = (sampling[i]).count(sampling_frequantli[i][a])

    plt.bar(sampling_frequantli[0], Y_frequency_histogram[0], label='EHMF' + str(1))
    plt.bar(sampling_frequantli[1], Y_frequency_histogram[1], label='EHMF' + str(2))
    plt.bar(sampling_frequantli[2], Y_frequency_histogram[2], label='EHMF' + str(3))
    plt.bar(sampling_frequantli[3], Y_frequency_histogram[3], label='EHMF' + str(4))
    plt.bar(sampling_frequantli[4], Y_frequency_histogram[4], label='EHMF' + str(5))

    plt.legend()
    plt.show()


def DifferenceOfEmpir(size):
    value = []
    x = viborka(size)
    for i in range(5):
        x[i].sort()

    for i in range(5):
        for j in range(size):
            x[i][j] = round(x[i][j], 0)

    for i in range(5):
        x[i] = list(set(x[i]))

    for i in range(5):
        value.append(len(x[i])/size)

    value = list(set(value))
    value.sort()
    if len(value) == 1 or len(value) == 0:
        DifferenceOfEmpir(size)
    max1 = value.pop(-1)
    min1 = value.pop(0)

    return max1 - min1


def sampleMean(volume):
    sum = 0
    sampleMean = []
    sampling = viborka(volume)
    for a in sampling:
        for i in a:
            sum += i
        sampleMean.append(sum / volume)
        sum = 0

    return sampleMean


def Dispersion(volume, numberOfMoment = 2):
    sum = 0
    dispersion = []
    sampleMeanOfDispersion = []
    sampling = viborka(volume)
    for a in sampling:
        for i in a:
            sum += i
        sampleMeanOfDispersion.append(sum / volume)
        sum = 0
    sum = 0
    index = 0
    for a in sampling:
        for i in a:
            sum += (i - sampleMeanOfDispersion[index])**numberOfMoment
        dispersion.append(sum / volume)
        index += 1
        sum = 0
    return dispersion


def randomSample(volume):
    for j in range(5):
        for z in range(volume):
            sampling[j].append(LaplaceDistribution(ar))
    return sampling


def likelyhood(volume):
    likelyhood = [1, 1, 1, 1, 1]
    for i in range(5):
        for j in range(volume):
            likelyhood[i] *= randomLaplace(randomSample(volume)[i][j], ar, bettaRandom)
    return likelyhood


def randomSampleMean(volume):
    sum = 0
    sampleMean = []
    sampling = randomSample(volume)
    for a in sampling:
        for i in a:
            sum += i
        sampleMean.append(sum / volume)
        sum = 0

    sam = (sampleMean[0]+sampleMean[1]+sampleMean[2]+sampleMean[3]+sampleMean[4])/5

    return sam


def approximateli_value(iterable, value):
    return min(iterable, key=lambda x: abs(x - value))


def DifferenceOfEmpirVlad(sample, value):
    sample.sort()
    proba = 0
    c = Counter(sample)
    c = sorted(c.items())
    c = dict(c)
    t = approximateli_value(c.keys(), value)
    if t >= value:
        for j in range(0, list(c.keys()).index(t)):
            proba += c[list(c.keys())[j]] / len(sample)
    else:
        for j in range(0, list(c.keys()).index(t)+1):
            proba += c[list(c.keys())[j]] / len(sample)
    return proba


def Smirnov(sample):
    res = 0
    for i in range(5):
        for j in range(4 - i):
            for k in range(len(sample[0])):
                if abs(DifferenceOfEmpir(sample[i], sample[i][k]) - DifferenceOfEmpir(sample[j], sample[j][k])) > res:
                    res = abs(DifferenceOfEmpir(sample[i], sample[i][k]) - DifferenceOfEmpir(sample[j], sample[j][k]))
    print(res)

k = list(range(228))
n = 228
p = 0.228
aa = []
for a in range(29,79):
    aa.append(((betainc(n-floor(a+1),1+floor(a+1),1-p)) - betainc(n-floor(a),1+floor(a),1-p))*10000)
print(aa)
