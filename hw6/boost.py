from tabulate import tabulate
import numpy

data_training = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw6train.txt").read().splitlines()]

data_testing = [numpy.array(map(int,x.rstrip().split())) for x in 
        open("hw6test.txt").read().splitlines()]

data_dictionary= open("hw6dictionary.txt").read().splitlines()

class container:
    feature = -1
    label = 0
    alpha = -1

    def __init__(self,a,f,l):
        self.alpha = a
        self.feature = f
        self.label = l

def class_pos(val): return 1 if val == 1 else -1

def list_tags(num, dix, cont_list):
    print "Words chosen by weak learners after",num,"rounds of testing:"
    print "\tSpam: ", ', '.join([dix[classi.feature] for classi in cont_list if
      classi.label < 0])
    print "\tNot Spam: ", ', '.join([dix[classi.feature] for classi in cont_list if
      classi.label > 0])
    print ''

def boost(data,weights):
    label = 0; min_err = float('inf'); feat_best = -1

    for i in xrange(len(data[0])-1):
        error = sum(weights[x] for x in xrange(len(data)) if 
          class_pos(data[x][i])!=data[x][-1])

        if (error < min_err) :
            feat_best = i
            min_err= error
            label = 1
        elif ( 1.0 - error < min_err ) :
            feat_best = i
            min_err = 1.0 - error
            label = -1
	
    alpha = 0.5*numpy.log(((1.0-min_err)/min_err))

    for i in xrange(len(data)):
        precomp = class_pos(data[i][feat_best]) if label > 0 else \
          class_pos(not data[i][feat_best])

        weights[i]=weights[i]*numpy.exp(-alpha*data[i][-1]*precomp)

    weights = map(lambda elm: elm/sum(weights), weights)
    return container(alpha,feat_best,label), weights

def classify(email, classifiers):
    return numpy.sign(sum(elm.alpha*class_pos(email[elm.feature]) 
      if elm.label > 0 else elm.alpha*class_pos(not email[elm.feature]) 
      for elm in classifiers))

def perform_tests(d_train, d_test):
    boost_counts = [3,7,10,15,20]
    score_table = [[" " for _ in xrange(3)] for _ in xrange(len(boost_counts)+1)]

    score_table[0][0] = "t"
    score_table[0][1] = "Training Error"
    score_table[0][2] = "Test Error"

    tests = [d_train, d_test]

    for idx,num in enumerate(boost_counts):
        score_table[idx + 1][0] = str(num)

        weights = [1.0/len(d_train)] * len(d_train)
        classi_cont_list = []

        for _ in xrange(num):
            res = boost(d_train,weights)
            classi_cont_list.append(res[0])
            weights = res[1]

        for idxx,data_set in enumerate(tests):
            score_table[idx + 1][idxx + 1] = str(sum(1 for email in data_set 
              if classify(email,classi_cont_list)!=email[-1])/float(len(data_set)))

        if num <= 10:
            list_tags(num,data_dictionary,classi_cont_list)

    return tabulate(score_table, headers="firstrow")

print ''; print perform_tests(data_training,data_testing)
