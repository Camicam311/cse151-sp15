from collections import defaultdict
from tabulate import tabulate
from numpy import dot
import numpy

data_training_a = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4atrain.txt").read().splitlines()]

data_testing_a = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4atest.txt").read().splitlines()]

data_training_b = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4btrain.txt").read().splitlines()]

data_testing_b = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw4btest.txt").read().splitlines()]

class wc:
    w = []
    c = 1

def build_voted_perceptron(data, passes = 1, sought_class=0):
    '''
    Generates dictionary of index -> wc objects for perceptron.
    '''

    wc_map = defaultdict(wc)
    wc_map[1].w = [0 for _ in xrange(len(data[0])-1)]

    m = 1

    # Should I restart m?
    for _ in xrange(passes):
        for feature in data:
            if int(process_label(feature[-1], sought_class) * 
              numpy.sum(dot(wc_map[m].w,feature[0:len(feature)-1]))) <= 0:

                wc_map[m+1].w = wc_map[m].w + dot(process_label(feature[-1], 
                  sought_class),feature[0:len(feature)-1])

                m += 1
                wc_map[m].c = 1
            else:
                wc_map[m].c += 1

    return wc_map

def process_label(label, classi):
    return 1 if label == classi else -1

def build_basic_perceptron(data, passes, sought_class=0):
    w = 0
    for _ in xrange(passes):
        for feature in data:
            if int(process_label(feature[-1],sought_class) * 
              numpy.sum(dot(w, feature[0:len(feature)-1])))<=0:

                w = w + dot(process_label(feature[-1],sought_class),
                  feature[0:len(feature)-1])
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

def classify_perceptron_set(data, classifier, function, sought_class=0):
    error_count = 0

    for feature in data:
        res = function(feature[:len(feature)-1], classifier)
        if res != process_label(feature[-1],sought_class):
            error_count += 1

    return str((error_count/float(len(data))) * 100) + "%"

def perform_tests(d_train, d_test):
    score_table = [[" " for _ in xrange(4)] for _ in xrange(4)]
    score_table[0][0] = "Pass"
    score_table[0][1] = "Basic"
    score_table[0][2] = "Voted"
    score_table[0][3] = "Avg'd"
    for num in xrange(3):
        score_table[num + 1][0] = str(num)
        bp = build_basic_perceptron(d_train, num+1)
        score_table[num + 1][1] = classify_perceptron_set(d_test, bp, 
          classify_basic_perceptron)

        vp = build_voted_perceptron(d_train, num+1)
        score_table[num + 1][2] = classify_perceptron_set(d_test, vp, 
          classify_voted_perceptron)

        score_table[num + 1][3] = classify_perceptron_set(d_test, vp, 
          classify_averaged_perceptron)


    return tabulate(score_table, headers="firstrow")

def calc_confusion_matrix():
    confusion_matrix = [[0 for _ in xrange(10)] for _ in xrange(11)]
    label_counts = [0 for _ in xrange(10)]
    multiclass = []

    for num in xrange(10):
        multiclass.append(build_basic_perceptron(data_training_b,1,num))

    for test_idx, test_datum in enumerate(data_testing_b):
        multi_res = [0 for _ in xrange(10)]

        label_counts[test_datum[-1]] += 1

        for idx,classifier in enumerate(multiclass):
            res=classify_basic_perceptron(test_datum[:len(test_datum)-1], 
              classifier)

            if res == 1:
                multi_res[idx] += res

        if numpy.sum(multi_res) == 1: #classified as singular thing
            for idx, elm in enumerate(multi_res):
                if elm == 1: 
                    confusion_matrix[idx][test_datum[-1]] += 1
        else: #Don't know
            confusion_matrix[10][test_datum[-1]] += 1

    for count, row in enumerate(confusion_matrix):
        for idx,elm in enumerate(row):
            row[idx] = float(elm) / float(label_counts[idx])
            row[idx] = str(row[idx])[:7]

    return tabulate(confusion_matrix)

def main():
    print "\n\n\tTraining Errors"
    print perform_tests(data_training_a, data_training_a)
    print "\n\n\tTest Errors"
    print perform_tests(data_training_a, data_testing_a)
    print "\n\n\tConfusion Matrix"
    print calc_confusion_matrix()

main()
