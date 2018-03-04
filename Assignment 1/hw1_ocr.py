import pandas as pd
import matplotlib.pyplot as mlt
import operator
import numpy as np
import random as rd


MAX = [1, 5, 10, 15, 20]
R = [1,5,10,20]
ETA = [0.01, 0.1, 1]
label_lenght = 26
feature_length = 128

output = open("ocr_output.txt", "w")


def format_inputs(filename):
    x = []
    y = []
    with open(filename) as train:
        ytemp = []
        xtemp = []
        for cnt, line in enumerate(train):
            line = line.split("\t")
            # print(cnt)
            if not line[0] == "\n":
                temp = list(line[1])
                temp = temp[2:]
                xtemp.append(temp)
                ytemp.append(line[2])
            else:
                x.append(list(xtemp))
                y.append(ytemp)
                ytemp = []
                xtemp = []
    print y
    # Converting everything to integers
    for i in range(len(y)):
        for j in range(len(y[i])):
            # print y[i][j]
            y[i][j] = ord(y[i][j]) - 96
    print y
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                x[i][j][k] = int(x[i][j][k])
    print len(x)
    print x
    # y = map(int, np.array(y[0]))
    # return np.array(x), y
    return x, y

def joint_feature(types, x, y):

    # x = [j for i in x for j in i]
    # print(x)
    if types == "unary":
        phi_x_y = [[0] * feature_length]*label_lenght
        for j in range(len(y)):
            # phi_x_y[i][y[i][j]] = np.add(phi_x_y[i][y[i][j]], x[i][j])
            # print phi_x_y[y[j]]
            phi_x_y[y[j]] = map(operator.add, phi_x_y[y[j]], x[j])
            # phi_x_y[i][y[i][j]] = x[i][j]
            # print phi_x_y[i][y[i][j]], i, y[i][j]
            # print x[i][j], i, j
            # print phi_x_y[0][4]
        phi_x_y = np.array(phi_x_y)
        # print phi_x_y
        features = np.ndarray.flatten(phi_x_y)
        features1 = list(features)
        # print "unary"
        return features1
    # elif types == "pair":
    #     w_unary = [0] * len(x[1][1]) * label_lenght
    #     w_pair = [0] * label_lenght * label_lenght
    #     print len(w_unary + w_pair)
    #     print "pair"
    #     return w_unary + w_pair   # , features
    # elif types == "triple":
    #     w_unary = [0] * len(x[1][1]) * label_lenght
    #     w_pair = [0] * label_lenght * label_lenght
    #     w_triple = [0] * label_lenght * label_lenght * label_lenght
    #     print len(w_unary + w_pair + w_triple)
    #     print "triple"
    #     return w_unary + w_pair + w_triple  # , features
    # else:
    #     w_unary = [0] * len(x[1][1]) * label_lenght
    #     w_pair = [0] * label_lenght * label_lenght
    #     w_triple = [0] * label_lenght * label_lenght * label_lenght
    #     w_quad = [0] * label_lenght * label_lenght * label_lenght * label_lenght
    #     print len(w_unary + w_pair + w_triple + w_quad)
    #     print "quad"
    #     return w_unary + w_pair + w_triple + w_quad  # , features

def score_function(weights, x, y, types):
    return np.dot(weights, joint_feature(types, x, y))


def rgs(x, phi_x_y, weights, R, types):
    dummy = 0
    # print len(x)
    # print len(phi_x_y)
    # print len(weights)
    y_best = []
    for i in range(len(x)):
        # print rd.randint(0,5)
        y_best.append(rd.randint(0,label_lenght-1))
    s_best = score_function(weights, x, y_best, types)
    for i in range(R):
        y_start = []
        for i in range(len(x)):
            # print rd.randint(0,5)
            y_start.append(rd.randint(0, label_lenght-1))
        temp1 = score_function(weights, x, y_start, types)
        temp2 = temp1
        for i in range(len(y_start)):
            for j in range(label_lenght-1):
                if y_start[i] != j:
                    temp2 = score_function(weights, x, y_start, types)
                if temp1 < temp2:
                    y_start[i] = j
                    temp1 = temp2
        if s_best < temp1:
            y_best = y_start
    return y_best
    # print "RGS"

def hamming(y_hat, y):
    dist = 0
    for ch1, ch2 in zip(y_hat, y):
        if ch1 == ch2: dist += 1
    return dist


def structuredpreceptron(stress_x_train, stress_y_train, types, max_iter, r, eta):
    print "Structured Perceptron"
    if types == "unary":
        weights = [0] * label_lenght * feature_length
    weights = np.array(weights)
    # print len(weights)
    # weights = joint_feature(types, stress_x_train, stress_y_train)
    updated_weights = [0] * len(weights)
    updated_weights = np.array(updated_weights)
    count = 0

    # y_hat = [[4, 0], [4, 4, 0, 1, 3], [4, 0, 3, 2], [4, 1, 3, 3], [4, 1, 2, 3], [4, 4, 1, 1, 3], [1, 0, 1, 1, 3, 3], [4, 1, 3, 3], [4, 4, 2, 3, 0, 2, 4, 0, 3], [4, 0, 4, 1, 3, 4, 0, 3, 3], [1, 3, 3]]

    accuracy_training = []
    for i in xrange(0, max_iter, 2):
        training_correct = training_total = 0
        for j, k in zip(stress_x_train, stress_y_train):
            if count == 1:
                break
            phi_x_y = joint_feature(types, j, k)
            str_k = list(map(str, k))
            str_k = ''.join(str_k)
            y_hat_1 = rgs(j, phi_x_y, updated_weights, r, types)
            str_y_hat = list(map(str, y_hat_1))
            str_y_hat = ''.join(str_y_hat)
            training_total += len(k)
            training_correct += hamming(str_y_hat, str_k)
            if hamming(str_y_hat, str_k):
                temp = map(operator.sub, phi_x_y, joint_feature(types, j, y_hat_1))
                temp = [eta * j for j in temp]
                updated_weights = map(operator.add, updated_weights, temp)

            # if weights == updated_weights:
            #     count += 2
        accuracy_training.append(training_correct/float(training_total) * 100)
        print "Training Total:", training_total, "Training Correct", training_correct
        print training_correct/float(training_total) * 100
    # mlt.plot(accuracy_training)
    # mlt.ylabel('Training Accuracy')
    # mlt.xlabel('Iterations in steps of 3: 1, 4, 7, 10....')
    # mlt.show()
    print "finished"
    return weights, training_total, training_correct


def testing(final_weights, ocr_x_train, ocr_y_train, types, r):
    count = 0
    testing_total = testing_correct = 0
    for j, k in zip(ocr_x_train, ocr_y_train):
        if count == 1:
            break
        phi_x_y = joint_feature(types, j, k)
        str_k = list(map(str, k))
        str_k = ''.join(str_k)
        y_hat_1 = rgs(j, phi_x_y, final_weights, r, types)
        str_y_hat = list(map(str, y_hat_1))
        str_y_hat = ''.join(str_y_hat)
        testing_total += len(k)
        testing_correct += hamming(str_y_hat, str_k)

    return testing_total, testing_correct


def main():

    ocr_train_file = "datasets/ocr_fold0_sm_train_sample.txt"
    ocr_test_file = "datasets/ocr_fold0_sm_test_sample.txt"
    ocr_x_train, ocr_y_train = format_inputs(ocr_train_file)
    ocr_x_test, ocr_y_test = format_inputs(ocr_test_file)

    for i in MAX:
        for j in R:
            for k in ETA:
                output.write("For Max iterations=%i Number of restarts=%i Learning Rate=%f \n" % (i, j, k))
                final_weights, training_total, training_correct = structuredpreceptron(ocr_x_train, ocr_y_train, "unary", i, j, k)
                testing_total, testing_correct = testing(final_weights, ocr_x_test, ocr_y_test, "unary", j)
                testing_accuracy = testing_correct/float(testing_total) * 100
                output.write("Testing Accuracy=%f \n\n\n\n" % (testing_accuracy))
                # print "Accuracy for testing = ", testing_correct/float(testing_total) * 100
                # print "For Max iterations=", i, " Number of restarts=", j, " Learning Rate=", k
    output.close()
    print("main")
    #output = open("stress_output.txt", "w")
    #final_weights = StructuredPreceptron()
    #output.close()

if __name__ == '__main__':
    main()