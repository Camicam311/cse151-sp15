from tabulate import tabulate
from numpy import dot
import numpy

data_training_a = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw5train.txt").read().splitlines()]

data_testing_a = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw5test.txt").read().splitlines()]

class wc:
    w = []
    c = 1

def num_common_substr(str_1, str_2, length):
    sub_list = set()
    for idx in xrange(len(str_1)-(length-1)):
        if str_1[idx:idx+length] in str_2:
            sub_list.add(str_1[idx:idx+length])

    return len(sub_list)

def process_label(label, classi):
    return 1 if label == classi else -1

def build_basic_perceptron(data, substr_len, sought_class=0):
    w = 0
    for feature in data:

        # What is my second string for searching for common substrings?
        if int(process_label(feature[-1],sought_class) * numpy.sum(dot(w, 
          num_common_substr(feature[0:len(feature)-1], None, substr_len))))<=0:

            # What is my second string for searching for common substrings?
            w = w + dot(process_label(feature[-1],sought_class),
              num_common_substr(feature[0:len(feature)-1], None, substr_len))
    return w

def classify_basic_perceptron(datum, classification_vector):
    return numpy.sign(numpy.dot(datum, classification_vector))

def classify_perceptron_set(data, classifier, function, sought_class=1):
    error_count = 0

    for feature in data:
        res = function(feature[:len(feature)-1], classifier)
        if res != process_label(feature[-1],sought_class):
            error_count += 1

    return str((error_count/float(len(data))) * 100) + "%"

def perform_tests(d_train, d_test):
    score_table = [[" " for _ in xrange(3)] for _ in xrange(3)]

    score_table[0][0] = "p"
    score_table[0][1] = "Training Error"
    score_table[0][2] = "Test Error"

    for idx,num in enumerate([3, 4]):
        score_table[idx + 1][0] = str(num)

        bp = build_basic_perceptron(d_train, num)

        score_table[idx + 1][1] = classify_perceptron_set(d_train, bp, 
          classify_basic_perceptron)
        score_table[idx + 1][2] = classify_perceptron_set(d_test, bp, 
          classify_basic_perceptron)


    return tabulate(score_table, headers="firstrow")

def main():
    print perform_tests(data_training_a, data_testing_a)

main()
