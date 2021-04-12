import random
import matplotlib.pyplot as plt
from scipy.special import comb as mc, betainc
from math import floor, e, log
import numpy as np

n = 100
p = 0.33
q = 1 - p

znacheniya_Y = []
znacheniaOfBinominalFunction = []
viborkaIsFive = [[], [], [], [],  []]
znacheniaViborkiIsFive = [[], [], [], [], []]
m = 0


def viborka(size):
    for j in range(5):
        for z in range(size):
            viborkaIsFive[j].append(int(Logarithmic(p)))
    return viborkaIsFive


def Logarithmic(p):
    logQInv = 1.0 / log(1.0 - p)

    V = random.uniform(0, 1)
    if V >= p:
        return 1.0
    U = random.uniform(0, 1)
    y = 1.0 - e ** (U / logQInv)
    if V > y:
        return 1.0
    if V <= y * y:
        return floor(1.0 + log(V) / log(y))
    return 2.0


def modelLogarithmic():
    Os_x = []
    Os_y = []
    for a in range(1, n):
        Os_x.append(Logarithmic(p))
    Os_x.sort()
    for a in Os_x:
        Os_y.append((-1/log(1 - p)) * (p ** a) / a)
    plt.plot(Os_x, Os_y)
    plt.grid(True)
    plt.show()


def modelTeoreticalLogarithmic():
    Os_x = []
    Os_y = []
    for a in range(1, n):
        Os_x.append(a)
    for a in Os_x:
        Os_y.append((-1/log(1 - p)) * (p ** a) / a)
    plt.plot(Os_x, Os_y, color='red')
    plt.grid(True)
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

    for i in range(n):
        FRX.append(i)
    FRX.sort()
    for a in FRX:
        FRY.append(1 + ((betainc(a, 1.0, p))/(np.log(1-p))))

    plt.step(ALLViborka[0], znacheniaViborkiIsFive[0], color='red', label='EDF' + str(1))
    plt.step(ALLViborka[1], znacheniaViborkiIsFive[1], color='orange', label='EDF' + str(2))
    plt.step(ALLViborka[2], znacheniaViborkiIsFive[2], color='yellow', label='EDF' + str(3))
    plt.step(ALLViborka[3], znacheniaViborkiIsFive[3], color='green', label='EDF' + str(4))
    plt.step(ALLViborka[4], znacheniaViborkiIsFive[4], color='blue', label='EDF' + str(5))
    plt.step(FRX, FRY, label='CDF', color='black')
    plt.xlim(0, 10)
    plt.grid(True)
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
    Y_frequency_polygon = [0, 0, 0, 0, 0]
    sampling = viborka(size)
    sampling_frequantli = sampling.copy()

    for i in range(5):
        sampling_frequantli[i] = list(set(sampling_frequantli[i]))
        sampling[i].sort()
        sampling_frequantli[i].sort()
        Y_frequency_polygon[i] = [0] * len(sampling_frequantli[i])

    for i in range(5):
        for a in range(len(sampling_frequantli[i])):
            Y_frequency_polygon[i][a] = (sampling[i]).count(sampling_frequantli[i][a])/len(sampling[i])

    plt.step(sampling_frequantli[0], Y_frequency_polygon[0], label='EPMF' + str(1))
    plt.step(sampling_frequantli[1], Y_frequency_polygon[1], label='EPMF' + str(2))
    plt.step(sampling_frequantli[2], Y_frequency_polygon[2], label='EPMF' + str(3))
    plt.step(sampling_frequantli[3], Y_frequency_polygon[3], label='EPMF' + str(4))
    plt.step(sampling_frequantli[4], Y_frequency_polygon[4], label='EPMF' + str(5))

    Ox = []
    Oy = []
    for i in range(1, size):
        Ox.append(i)
    Ox.sort()
    for a in Ox:
        Oy.append((-1/log(1 - p)) * (p ** a) / a)

    plt.plot(Ox, Oy, label='PMF', color='black')
    plt.xlim(0, 20)
    plt.grid(True)
    plt.legend()
    plt.show()


def frequency_histogram(size):
    Y_frequency_histogram = [0, 0, 0, 0, 0]
    sampling = viborka(size)
    sampling_frequantli = sampling.copy()

    for i in range(5):
        sampling_frequantli[i] = list(set(sampling_frequantli[i]))
        sampling[i].sort()
        sampling_frequantli[i].sort()
        Y_frequency_histogram[i] = [0] * len(sampling_frequantli[i])

    for i in range(5):
        for a in range(len(sampling_frequantli[i])):
            Y_frequency_histogram[i][a] = (sampling[i]).count(sampling_frequantli[i][a])

    plt.bar(sampling_frequantli[0], Y_frequency_histogram[0])
    plt.bar(sampling_frequantli[1], Y_frequency_histogram[1])
    plt.bar(sampling_frequantli[2], Y_frequency_histogram[2])
    plt.bar(sampling_frequantli[3], Y_frequency_histogram[3])
    plt.bar(sampling_frequantli[4], Y_frequency_histogram[4])
    plt.grid(True)
    plt.show()


def Max_Difference(volume):
    MaxDifference = 0
    x = viborka(volume)
    for i in range(5):
        for j in range(i + 1, 5):
            MaxDifference = max(MaxDifference, abs(max(x[i]) - max(x[j])))
        print(MaxDifference)

# , 100, 1000, 100000
for size in [5, 10, 100, 1000, 100000]:
   (frequency_histogram(size))

# print(viborka(10))
# plt.show()
