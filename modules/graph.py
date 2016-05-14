from random import shuffle
from ID3 import *
from operator import xor
from parse import parse
import matplotlib.pyplot as plt
import os.path
from pruning import *
import copy
import random
import numpy as np 

# NOTE: these functions are just for your reference, you will NOT be graded on their output
# so you can feel free to implement them as you choose, or not implement them at all if you want
# to use an entirely different method for graphing

def get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, pct):
    '''
    get_graph_accuracy_partial - Given a training set, attribute metadata, validation set, numerical splits count, and percentage,
    this function will return the validation accuracy of a specified (percentage) portion of the trainging setself.
    '''
    
    shuffle(train_set)
    train_subset = copy.deepcopy(train_set[:int(pct*len(train_set))])
    thisSplits = copy.deepcopy(numerical_splits_count)
    #print len(train_subset)
    tree = ID3(train_subset, attribute_metadata, thisSplits, depth)
    if prune != False:
        pruning_set, _ = parse(prune, False)
        reduced_error_pruning(tree,train_set,pruning_set)
    accuracy = validation_accuracy(tree,validate_set)
    #print accuracy
    return accuracy

def get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, iterations, pcts):
    '''
    Given a training set, attribute metadata, validation set, numerical splits count, iterations, and percentages,
    this function will return an array of the averaged graph accuracy partials based off the number of iterations.
    '''
    accuracies = []

    for pct in pcts:
        iters = []
        for it in range(0,iterations):
            iters.append(get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, pct, prune))
        avg = sum(iters)/float(iterations)
        print avg
        accuracies.append(avg)
    #print accuracies
    return accuracies

# get_graph will plot the points of the results from get_graph_data and return a graph
def get_graph(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, lower, upper, increment):
    '''
    get_graph - Given a training set, attribute metadata, validation set, numerical splits count, depth, iterations, lower(range),
    upper(range), and increment, this function will graph the results from get_graph_data in reference to the drange
    percentages of the data.
    '''
    #print numerical_splits_count
    x = lower
    pcts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    accuracies = get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pcts, prune)
    #print accuracies
    plt.plot(pcts, accuracies)
    plt.show()