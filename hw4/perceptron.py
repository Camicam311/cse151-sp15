from collections import defaultdict
from numpy import dot
import numpy

td = [[4,0,1],[1,1,-1],[0,1,-1],[-2,-2,1]]

class wc:
    w = []
    c = float('-inf')

def calc_perceptron(data):
    '''
    Generates dictionary of index -> wc objects for perceptron.
    '''

    wc_map = defaultdict(wc)

    w = 0
    m = 1

    for idx, feature in enumerate(data):
        if int(feature[-1] * numpy.sum(dot(w,feature[0:len(feature)-1]))) <= 0:
            w = w + dot(feature[-1],feature[0:len(feature)-1])
            m += 1
            wc_map[m].w = w
            wc_map[m].c = 1
        else:
            wc_map[m].c += 1

    return wc_map

def print_wc_map(wcm):
    for idx in wcm:
        print "w["+str(idx)+"], c["+str(idx)+"]:",wcm[idx].w,",",wcm[idx].c

basic_perceptron = calc_perceptron(td)
print_wc_map(basic_perceptron)


