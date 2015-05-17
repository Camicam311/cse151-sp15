from collections import namedtuple, defaultdict
from pprint import pprint
from math import log

data_train = open('hw3train.txt').readlines()
data_test = open('hw3test.txt').readlines()

Flower = namedtuple('Flower', ['p_width', 'p_length', 's_width', 's_length', 
                               'species'])

flowers = [Flower(*elm.split()) for elm in data_train]

def is_impure(ls):
    for num in [x[-1] for x in ls]:
        if num != ls[0][-1]:
            return True
    return False

def is_pure(ls):
    return not is_impure(ls)

def entropy(ls,idx=-1):
    label_counts = {}
    for label in set([lab[idx] for lab in ls]):
        label_counts[label] = [x[idx] for x in ls].count(label)
  
    return sum([-1 * float(x)/len(ls) * log(float(x)/len(ls), 2) for x in label_counts.values()])

def info_gain(attr, data):
    attr_counts = {}
    ent_sub = 0.0

    ms = set([at[attr] for at in data])
    aa = [a[attr] for a in data]
    for x in ms:
        attr_counts[x] = aa.count(x)

    #for atr in set([at[attr] for at in data]):
    #    attr_counts[atr] = [a[attr] for a in data].count(str(attr))

    for val in attr_counts.keys():
        p = attr_counts[val]/float(len(data))
        ent_sub += p * entropy([elm for elm in data if elm[attr] == val])

    return entropy(data) - ent_sub

def best_feature(data):
    best_info = -1.0
    best_attr = -1

    for x in range(4):
        res = info_gain(x,data)
        if res > best_info:
            best_info = res
            best_attr = x

    return best_attr

#DONT TRUST THIS FUNCTION OR ITS USAGE
def split_data(data,best,x):
    res = []
    for v in data:
        if v[best] == x:
            r_rec = v[:best]
            l = [r for r in r_rec]
            l.extend(v[best+1:])
            res.append(l)
    return res

def build_tree(data):
    #if not data or len(data) == 1:
    #    return 1 # don't leave this here
    if is_pure(data):
        return int(data[0][-1])
    best = best_feature(data)
    print "If feature[",best+1,"] <=","??????"
    feat_values = [x[best] for x in data]
    uniq = set(feat_values)
    tree = defaultdict(lambda:{}) #### Not sure????
    for x in uniq:
        tree[best][str(x)] = build_tree(split_data(data, best, x))


    #print "feat idx",best
    #print info_gain(best,data)
    #a = set([at[best] for at in data])
    #print "a,",a
    #for x in set([at[best] for at in data]):
        #tree[best][str(x)] = build_tree([dat for dat in data if dat[best] == x])
        #this is the one you were useing#tree[best][str(x)] = build_tree([dat for dat in data if dat[best] == x])
        #print build_tree([dat for dat in data if dat[best] == x])
    return tree


x = build_tree(flowers)

pprint(x)
#print "Len",len(x)
for y in x:
    #print "IF feat",y+1,"<="
    for a in x[y]:
        #print a
        pass
''' Bonus code saving for later
    h = defaultdict(lambda: {})
    for x in data:
        for field in x._fields:
            if getattr(x,field) not in h[field]:
                h[field][getattr(x,field)] = 1
            else:
                h[field][getattr(x,field)] += 1
    print h
'''
