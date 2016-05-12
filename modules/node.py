# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 nodes if numeric, and a dictionary (key=attribute value, value=node) if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is the output label (0 or 1 for
#	the homework data set) if there are no other attributes
#       to split on or the data is homogenous
#	if there is a decision attribute, your implementation can choose set label to None
#	or to the class mode of the examples
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on
from copy import deepcopy

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        #child dict is empty --> False so we have no children and are a lead node with label
        if not self.children:
            print len(self.children)
            print "THERE"
            return self.label

        print len(instance)
        # decision_index = self.decision_attribute
        # print "HHHHHHHH"
        # truth = instance[decision_index]

        # if self.is_nominal:
        #     print "it is nominal"
        #     #something here with children and splitting
        #     tempKey = self.decision_attribute
        #     print "============"
        #     print tempKey
        #     print "============"
        #     for key, value in self.children.iteritems():
        #         if tempKey == key:
        #             print key
        #             print value
        #             print "HERE"
        #             self.classify(value)
        
        # print "boring"
        # print self.decision_attribute
        # tempIndex = deepcopy(self.decision_attribute)
        # tempValue = deepcopy(instance[tempIndex])

        # we are at the case where we have children and the decision attribute is numeric
        if  error < self.splitting_value:
            print "HERE"
            self.classify(self.children[0])
        else:
            print "wait"
            self.classify(self.children[1])








        

    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        IMPLEMENTING THIS FUNCTION IS OPTIONAL
        '''
        # Your code here
        pass


    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        pass