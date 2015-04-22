import math
import sys
import operator

data_training = [x.rstrip().split() for x in open("hw2train.txt").read().splitlines()]
data_training_map = {}

def distance(comp_a, comp_b):
    distance = 0

    if len(comp_a) != len(comp_b):
        print "Oh shit son, you're fucked. len's aren't the same, abandon ship."
        sys.exit()

    for (x,y) in zip(comp_a, comp_b):
        distance += pow((int(x)-int(y)),2)

    return math.sqrt(distance)

def get_neighbors(input_, num_neighbors):
    ls = [(distance(input_[:len(input_)-1],data[:len(data)-1]), data) for data in data_training]
    ls.sort(key=operator.itemgetter(0))
    return ls[:num_neighbors]

def calc_error(samples, actual):
    sum_ = 0.0
    for sample in samples:
        if int(sample) == int(actual):
            sum_ += 1.0
    return 1- (sum_/len(samples))

def build_k_classifiers(k):
    for data in data_training:
        sample = [(z[-1]) for (x,z) in get_neighbors(data,k)]
        print "Expected: " + str(data[-1]) + " Sample Results: " + str(sample) + " ERR: " + str(100 * calc_error(sample, data[-1]))[:5]+"%"

build_k_classifiers(3)

        #print "CLASSIFICATION LABEL: " + y[-1]
        #for x in range(28):
        #print y[x*28:x*28+28]
        #print "#"*28*6
        #data_training_map.setdefault(y[-1],[]).append(y[0:len(y)-1])

