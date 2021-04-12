from math import e, sqrt

exp = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3]
exp.sort()


expexp = list(set(exp))

ex = 0
for a in [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3]:
    ex += a


def Dispersion(volume = 20, numberOfMoment = 2):
    sum = 0
    dispersion = []
    sampleMeanOfDispersion = []
    sampling = exp
    for i in sampling:
        sum += i
    sampleMeanOfDispersion.append(sum / volume)

    sum = 0
    index = 0

    for i in sampling:
        sum += (i - sampleMeanOfDispersion[index])**numberOfMoment
    dispersion.append(sum / volume)
    index += 1
    sum = 0
    return dispersion
disp = Dispersion().pop()


fx = 1/(-((ex/20)-sqrt(3)*sqrt(disp))+(ex/20)+sqrt(3)*sqrt(disp))
i = 8
print(20*fx*(expexp[i+1]-expexp[i]))
