import random
import matplotlib.pyplot as plt
from scipy.special import comb as mc
from math import floor
from collections import Counter

n = 228
p = 0.228
q = 1 - p

znacheniya_Y = []
znacheniaOfBinominalFunction = []
viborkaIsFive = [[], [], [], [],  []]
znacheniaViborkiIsFive = [[], [], [], [], []]
m = 0


def viborka(size):
    for j in range(5):
        for z in range(size):
            viborkaIsFive[j].append(int(Binomka(p, n)))
    return viborkaIsFive


def choose(N, K):
    if K == 0:
        return 1
    else:
        return (N * choose(N - 1, K - 1)) // K


def Binomka(p, n):
    c = p/(1-p)
    s = (1-p)**n
    r = s
    k = 0
    a = random.uniform(0, 1)
    while a > s:
        k += 1
        r = r * c * (n - k + 1)/k
        s = s + r
    return k


def modelBinomka():
    Os_x = []
    Os_y = []
    for a in range(n):
        Os_x.append(Binomka(p, n))
    Os_x.sort()
    for a in Os_x:
        Os_y.append(choose(n, a) * (p ** a) * (q ** (n - a)))
    plt.bar(Os_x, Os_y)


def modelTeoreticalBinomka():
    Os_x = []
    Os_y = []
    for a in range(n):
        Os_x.append(a)
    for a in Os_x:
        Os_y.append(choose(n, a) * (p ** a) * (q ** (n - a)))
    plt.plot(Os_x, Os_y, color='red')


def GraphEmpir(size):
    ALLViborka = viborka(size)
    MaxValue = 0
    znacheniaViborkiIsFive = [[], [], [], [], []]
    FRX = []
    FRY = []
    sum = 0
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

    for i in range(1, MaxValue):
        FRX.append(i)
        for j in range(i):
            sum += mc(n, j)*(p**j)*(q**(n-j))
        FRY.append(sum)
        sum = 0
    print(ALLViborka)
    plt.step(ALLViborka[0], znacheniaViborkiIsFive[0], color='red', label='EDF' + str(0))
    plt.step(ALLViborka[1], znacheniaViborkiIsFive[1], color='orange', label='EDF' + str(1))
    plt.step(ALLViborka[2], znacheniaViborkiIsFive[2], color='yellow', label='EDF' + str(2))
    plt.step(ALLViborka[3], znacheniaViborkiIsFive[3], color='green', label='EDF' + str(3))
    plt.step(ALLViborka[4], znacheniaViborkiIsFive[4], color='blue', label='EDF' + str(4))
    plt.stem(FRX, FRY, label='CDF'+str(5))
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
    for i in range(size):
        Ox.append(Binomka(p, n))
    Ox.sort()
    for a in Ox:
        Oy.append(choose(n, a) * (p ** a) * (q ** (n - a)))

    plt.plot(Ox, Oy, label='EMF', color='black')
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
    plt.show()


def DifferenceOfEmpir(size):
    value = []
    x = viborka(size)
    for i in range(5):
        x[i].sort()

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




for size in [5, 10, 100, 1000, 10000]:
   print(GraphEmpir(size))










