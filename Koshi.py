import random
from math import log, e, floor
import matplotlib.pyplot as plt
import numpy as np
import pylab

n = 1488
gamma = 1.0
x0 = 0

sampling = [[], [], [], [], []]
znacheniya_Y = []
znacheniaOfBinominalFunction = []
viborkaIsFive = [[], [], [], [],  []]
znacheniaViborkiIsFive = [[], [], [], [], []]


def KoshiDistribution():
    a1 = (random.uniform(0, 1))
    a2 = (random.uniform(0, 1))
    betta = 2 * a1 - 1
    d = betta * betta + a2 * a2
    while d > 1:
        a1 = (random.uniform(0, 1))
        a2 = (random.uniform(0, 1))
        betta = 2 * a1 - 1
        d = betta * betta + a2 * a2
    return a2/betta


def modelTheoreticalKoshiDistribution():
    Os_x = []
    Os_y = []
    for a in range(-10, 10):
        Os_x.append(a)
    for a in Os_x:
        Os_y.append(1/(np.pi * gamma * (1 + ((a - x0)/gamma) ** 2)))
    plt.plot(Os_x, Os_y, label='Teoretical', color='red')
    plt.legend()
    # plt.show()


def viborka(size):
    for j in range(5):
        for a in range(size):
            viborkaIsFive[j].append(KoshiDistribution())
    return viborkaIsFive


def modelGraphOfKoshi(n):
    counter = 0
    Os_of_x = []
    Os_of_y = []
    while counter <= n:
        Os_of_x.append(KoshiDistribution())
        counter += 1
    Os_of_x.sort()
    for a in Os_of_x:
        Os_of_y.append(1/(np.pi * gamma * (1 + ((a - x0)/gamma) ** 2)))
    plt.bar(Os_of_x, Os_of_y, label='Sample')
    pylab.xlim(-10, 10)
    plt.legend()
    # plt.show()


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

    for i in range(-5, 5):
        FRX.append(i)
    for a in FRX:
        y = ((1/np.pi)*(np.arctan((a - x0)/gamma))+0.5)
        FRY.append(y)

    plt.plot(FRX, FRY, color='black', label='CDF' + str(5))
    pylab.xlim(-5, 5)
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
    for i in range(-5, 5):
        Os_x.append(i)
    for a in Os_x:
        Os_y.append(1/(np.pi * gamma * (1 + ((a - x0)/gamma) ** 2)))

    plt.plot(Os_x, Os_y, label='PMF')
    pylab.xlim(-5, 5)
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
    pylab.xlim(-5, 5)
    plt.legend()
    plt.show()


def Max_Difference(size):
    MaxDifference = 0
    x = viborka(size)
    for i in range(5):
        for j in range(i + 1, 5):
            MaxDifference = max(MaxDifference, abs(max(x[i]) - max(x[j])))
        print(MaxDifference)


# , 100, 1000, 100000
for size in [5, 10, 100, 1000, 100000]:
    print(frequency_histogram(size))
