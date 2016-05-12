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
        self.depth = 0

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''

        if self.label is not None:
            return self.label

        check_val = instance[self.decision_attribute]

        # If nominal
        if self.is_nominal:
            return self.children[check_val].classify(instance)

        # If numerical
        else:
            if check_val < self.splitting_value:
                return self.children[0].classify(instance)
            else:
                return self.children[1].classify(instance)

    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        IMPLEMENTING THIS FUNCTION IS OPTIONAL
        '''
        if self.label is not None:
            print "\n" + ("  " * indent) + "Classify: " + str(self.label), self
            return

        if self.is_nominal:
            print "\n" + ("  " * indent) + "Nominal: ", self, self.children
            for key, child in self.children.iteritems():
                child.print_tree(indent+1)
        else:
            print "\n" + ("  " * indent) + "Numerical: ", self, self.children
            for child in self.children:
                child.print_tree(indent+1)


    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        pass
