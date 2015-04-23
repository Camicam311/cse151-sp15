import os

for num in [1,3,5,11,16,21]:
    os.system("python hw2.py " + str(num) + " > k" + str(num) + "_valid")
