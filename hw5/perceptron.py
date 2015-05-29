from tabulate import tabulate
import numpy

data_training_a = [numpy.array(map(str,x.rstrip().split())) for x in 
        open("hw5train.txt").read().splitlines()]

data_testing_a = [numpy.array(map(str,x.rstrip().split())) for x in 
        open("hw5test.txt").read().splitlines()]

def build_substring_matrix(data, length):
    return [[num_common_substr(data[x][0],data[y][0],length) for y in 
      xrange(len(data))] for x in xrange(len(data))]

def num_common_substr(str_1, str_2, length):
    sub_list = set()
    for idx in xrange(len(str_1)-(length-1)):
        if str_1[idx:idx+length] in str_2:
            sub_list.add(str_1[idx:idx+length])

    return len(sub_list)

def build_basic_perceptron(data, substr_len, sim_matrix, sought_class=1):
    w = [(0,0,0)]
    for idx, (protien, classification) in enumerate(data):
        dotio = reduce(lambda acc,(pro,clas,row):acc+clas*sim_matrix[row][idx],w,0)
        if int(classification) * dotio <= 0:
            w += [(protien, int(classification), idx)]
    return w

def classify_basic_perceptron(datum, train_data, w, k):
    x = reduce(lambda acc,(pro,clas,row):
      acc+clas*num_common_substr(data_training_a[row][0],datum, k),w,0)

    return numpy.sign(x)

def classify_perceptron_set(data, classifier, function, num, sought_class=1):
    error_count = 0

    for feature in data:
        res = function(feature[0],data,classifier, num)
        if res != int(feature[-1]):
            error_count += 1

    return str((error_count/float(len(data))) * 100) + "%"

def perform_tests(d_train, d_test):
    score_table = [[" " for _ in xrange(3)] for _ in xrange(3)]

    score_table[0][0] = "p"
    score_table[0][1] = "Training Error"
    score_table[0][2] = "Test Error"

    for idx,num in enumerate([3, 4]):
        score_table[idx + 1][0] = str(num)

        sim_matrix = build_substring_matrix(d_train, num)

        bp = build_basic_perceptron(d_train, num, sim_matrix)

        score_table[idx + 1][1] = classify_perceptron_set(d_train, bp, 
          classify_basic_perceptron, num)
        score_table[idx + 1][2] = classify_perceptron_set(d_test, bp, 
          classify_basic_perceptron, num)

    return tabulate(score_table, headers="firstrow")

def main():
    print "\n", perform_tests(data_training_a, data_testing_a)

main()
