from collections import defaultdict
from numpy import dot
import numpy

td = [[4,0,1],[1,1,-1],[0,1,-1],[-2,-2,1]]

data_training_a = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4atrain.txt").read().splitlines()]

data_testing_a = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4atest.txt").read().splitlines()]

data_training_b = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4btrain.txt").read().splitlines()]

class wc:
    w = []
    c = 1

def build_voted_perceptron(data, passes = 1):
    '''
    Generates dictionary of index -> wc objects for perceptron.
    '''

    wc_map = defaultdict(wc)
    wc_map[1].w = [0 for _ in xrange(len(data[0])-1)]

    m = 1

    # Should I restart m?
    for _ in xrange(passes):
        for feature in data:
            if int(process_label(feature[-1]) * numpy.sum(dot(wc_map[m].w,
                feature[0:len(feature)-1]))) <= 0:

                wc_map[m+1].w = wc_map[m].w + dot(process_label(feature[-1]),
                        feature[0:len(feature)-1])

                m += 1
                wc_map[m].c = 1
            else:
                wc_map[m].c += 1

    return wc_map

def process_label(label):
    return -1 if label == 0 else 1

def build_basic_perceptron(data, passes = 1):
    w = 0

    for _ in xrange(passes):
        for feature in data:
            if int(process_label(feature[-1]) * numpy.sum(dot(w,
                feature[0:len(feature)-1])))<=0:

                w = w + dot(process_label(feature[-1]),feature[0:len(feature)-1])
    return w

def classify_basic_perceptron(datum, classification_vector):
    return numpy.sign(numpy.dot(datum, classification_vector))

def classify_voted_perceptron(datum, classifier):
    s = 0
    for x in classifier:
        s += classifier[x].c * numpy.sign(numpy.dot(classifier[x].w,datum))

    return numpy.sign(s)

def classify_averaged_perceptron(datum, classifier):
    comp = [0 for _ in xrange(len(datum))]
    for x in classifier:
        comp = numpy.add(comp, classifier[x].c * classifier[x].w)
    return numpy.sign(numpy.dot(comp,datum))

def print_wc_map(wcm):
    print "Voted Perceptron Classification Vectors:"
    for idx in wcm:
        print "\tw["+str(idx)+"], c["+str(idx)+"]:",wcm[idx].w,",",wcm[idx].c

def classify_perceptron_set(data, classifier, function):
    error_count = 0

    for feature in data:
        res = function(feature[:len(feature)-1], classifier)
        if res != process_label(feature[-1]):
            error_count += 1

    print "\t\tError %:", str((error_count/float(len(data))) * 100) + "%"



"""
Begin main script shit.
"""
def perform_tests(d_train, d_test):
    for num in xrange(3):
        print "\n",num + 1,"Pass"
        print "\n\t","Basic"
        bp = build_basic_perceptron(d_train, num+1)
        classify_perceptron_set(d_test, bp, classify_basic_perceptron)

        print "\n\t","Voted"
        vp = build_voted_perceptron(d_train, num+1)
        classify_perceptron_set(d_test, vp, classify_voted_perceptron)

        print "\n\t","Averaged"
        classify_perceptron_set(d_test, vp, classify_averaged_perceptron)

print "Training Errors"
perform_tests(data_training_a, data_training_a)
print "\nTest Errors"
perform_tests(data_training_a, data_testing_a)
