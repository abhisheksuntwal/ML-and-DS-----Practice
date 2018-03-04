import pandas as pd
import operator
import numpy as np
import random as rd
import time


start_time = time.time()
MAX = [1, 5, 10, 15, 20]
R = [1,5,10,20]
ETA = [0.01, 0.1, 1]
label_lenght = 5
feature_length = 81

# output = open("stress_output.txt", "w")

def read_data(file_name):
    df = pd.read_table(file_name, sep="\t", header=None)
    x = df[:][1]
    x = x.str.strip('im')
    df[:][1] = x
    print df[0:5]

    return df

def structured_perceptron(train):
    print "structured_perceptron"

    return 0


if __name__ == '__main__':

    stress_train_file = "datasets/nettalk_stress_train_sample.txt"
    train = read_data(stress_train_file)


    weights = structured_perceptron(train)

