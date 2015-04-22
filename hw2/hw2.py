import math
import sys
import operator
from xtermcolor import colorize
import numpy
from scipy import stats


data_training = [numpy.array(map(int,x.rstrip().split())) for x in open("hw2train.txt").read().splitlines()]
data_training_map = {}

''' 
Computes Euclidian distance.
Returns: int
'''
def distance(comp_a, comp_b):
    return numpy.linalg.norm(comp_a-comp_b)

'''
Finds num_neighbors for input_.
Returns: [(distance, [neighbor_vector]), (distance, [neighbor_vector]), ...] of size num_neighbors
'''
def get_neighbors(input_, num_neighbors):
    ls = [(distance(input_[:len(input_)-1],data[:len(data)-1]), data) for data in data_training]
    ls.sort(key=operator.itemgetter(0))
    return ls[:num_neighbors]

''' 
Calculates error.
Returns: int -- 0.3333, 0.6666, etc
'''
def calc_error(samples, actual):
    sum_ = 0.0
    for sample in samples:
        if sample == actual:
            sum_ += 1.0
    return 1- (sum_/len(samples))

'''
Builds k classifiers
'''
global total_err
total_err = 0.0

def build_k_classifiers(k, input_list):
    global total_err
    for data in input_list:
        res = get_neighbors(data,k)
        sample = [(z[-1]) for (x,z) in res]
        actual = data[-1]
        #print sample
        #print actual
        prediction = stats.mode(sample)[0][0]
        if prediction != actual:
            error_percentage = calc_error(sample,actual)
            print "Predicted: " + str(prediction) + " " + "Actual: " + str(actual)+ " Sample Results: " + str(sample) + " ERR: " + str(100 * error_percentage)[:5]+"%"
            total_err += 1
'''
Pretty prints vector as image to terminal for debugging.
'''
def pretty_print_vector(vector):
    for res in vector:
        for x in range(28):
            ss = ''
            for y in res[x*28:x*28+28]:
                ss += colorize("@",int(y))
            print ss


build_k_classifiers(16, data_training)
print ""
print "Total errors: " + str(total_err) + "/" + str(len(data_training)) + " -- " + str((total_err/len(data_training))*100)[:5] + "%"
