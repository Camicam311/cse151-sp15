from collections import defaultdict
from numpy import dot
import numpy

td = [[4,0,1],[1,1,-1],[0,1,-1],[-2,-2,1]]

class wc:
    w = []
    c = 1

def build_voted_perceptron(data):
    '''
    Generates dictionary of index -> wc objects for perceptron.
    '''

    wc_map = defaultdict(wc)
    wc_map[1].w = [0 for _ in xrange(len(data[0])-1)]

    m = 1

    for feature in data:
        if int(feature[-1] * numpy.sum(dot(wc_map[m].w,
            feature[0:len(feature)-1]))) <= 0:

            wc_map[m+1].w = wc_map[m].w + dot(feature[-1],
                    feature[0:len(feature)-1])

            m += 1
            wc_map[m].c = 1
        else:
            wc_map[m].c += 1

    return wc_map

def build_basic_perceptron(data):
    w = 0

    for feature in data:
        if int(feature[-1] * numpy.sum(dot(w,feature[0:len(feature)-1]))) <= 0:
            w = w + dot(feature[-1],feature[0:len(feature)-1])

    return w

def classify_basic_perceptron(datum, classification_vector):
    return numpy.sign(numpy.dot(datum, classification_vector))

def classify_voted_perceptron(datum, classifier):
    #s = ""
    s = 0
    for x in classifier:
        s += classifier[x].c * numpy.sign(numpy.dot(classifier[x].w,datum))
    return numpy.sign(s)

def classify_averaged_perceptron(datum, classifier):
    comp = [0 for _ in xrange(len(datum) - 1)]
    for x in classifier:
        comp = numpy.add(comp, classifier[x].c * classifier[x].w)
    return numpy.sign(numpy.dot(comp,datum))

def print_wc_map(wcm):
    print "Voted Perceptron Classification Vectors:"
    for idx in wcm:
        print "\tw["+str(idx)+"], c["+str(idx)+"]:",wcm[idx].w,",",wcm[idx].c


"""
Begin main script shit.
"""

bp = build_basic_perceptron(td)
print "Basic Perceptron Classification Vector: ",bp

vp = build_voted_perceptron(td)
print_wc_map(vp)

print "Basic Classification: ",classify_basic_perceptron([4,0],bp)
print "Voted Classification: ",classify_voted_perceptron([4,0],vp)
print "Avg'd Classification: ",classify_averaged_perceptron([4,0],vp)

