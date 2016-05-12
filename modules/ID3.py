import math
from node import Node
import sys
from copy import deepcopy

def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================

    '''
    print "==== DEPTH ====", depth
    root_node = Node()
    queue = [[data_set, root_node]]

    # Loop through queue
    for [data, node] in queue:
        print "Queue:", len(queue)
        # If data is homogenous, set as leaf node
        if len(data) == 0:
            continue

        homogenous_value = check_homogenous(data)
        if homogenous_value is not None:
            print 'homo! depth=', node.depth
            node.label = homogenous_value
            continue

        # If node.depth == depth
        if node.depth == depth:
            print "Depth:", node.depth, data, mode(data)
            node.label = mode(data)
            continue

        # If data is not homogenous, split and create new nodes to add to queue
        best_attribute = pick_best_attribute(data, attribute_metadata, numerical_splits_count)

        # If split_counts == 0...
        if best_attribute == False:
            print "=== SPLIT COUNT FALSE ==="
            node.label = mode(data)
            continue

        # Create new nodes
        [best_i, split_value] = best_attribute
        if best_i == 0:
            print "============= UH OH ================"
        numerical_splits_count[best_i] -= 1
        node.decision_attribute = best_i

        # Check if best_attribute is nominal
        if split_value == False:
            node.is_nominal = True

            # Set node.children to child nodes
            nominal_split = split_on_nominal(data, best_i)
            for [attr_val, new_data] in nominal_split.iteritems():
                new_node = Node()
                new_node.depth = node.depth+1
                node.children[attr_val] = new_node

                # Append child nodes to queue
                queue.append([new_data, new_node])

        # If best_attribute is numerical
        else:
            node.is_nominal  = False
            node.split_value = split_value

            numerical_split = split_on_numerical(data, best_i, split_value)
            left_node  = Node()
            right_node = Node()
            left_node.depth  = node.depth +1
            right_node.depth = node.depth +1
            left_data  = numerical_split[0]
            right_data = numerical_split[1]


            # Set node.children to child nodes
            node.children = [left_node, right_node]


            # Append child nodes to queue
            queue.append([left_data,  left_node])
            queue.append([right_data, right_node])

    root_node.print_tree()
    return root_node

def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the output value (index 0) is the same for all examples in the the data_set, if so return that output value, otherwise return None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
     '''
    first_val = data_set[0]
    #value of first element in data set
    value = first_val[0]
    #for loop that exits and returns None any time a value isn't equal to initial value
    for e in data_set:
        if e[0] != value:
            return None
    return value

# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    # need to go through all the attributes in attribute metadata
    # if there are no numeric attributes the second value of output is False

    # use this number for looping through the metadata to find numeric or not
    num_attrs = len(attribute_metadata)

    finalArray = []

    #gainResults = []
    for e in range(1, num_attrs):

        # tuple with gain ratio and False if nominal and split value if numeric
        gainResult = []
        gainResult2 = []

        # name_nominal is a list with the first element being the attribute name and the second element
        # being a boolean True meaning the attribute is nominal False meaning is numeric
        name_nominal = attribute_metadata[e].values()

        # if this attribute is a nominal attr evaluate with gain_ratio_nominal
        if name_nominal[0] == True:

            tempGainRatio = gain_ratio_nominal(data_set, e)

            #final result here is [gain ratio, False]
            gainResult.append(tempGainRatio)
            gainResult.append(False)


            # looks like (attributeName, {gain ratio, False})
            # name_nominal[1] is the attribute name
            finalArray.append((name_nominal[1], gainResult))

        else:
            tempGainRatio2 = gain_ratio_numeric(data_set, e, 1)

            # below is gain_ratio
            gainResult2.append(tempGainRatio2[0])
            gainResult2.append(tempGainRatio2[1])



            # looks like (attributeName, {gain ratio, split value})

            finalArray.append((name_nominal[1], gainResult2))


    # need to cycle through attribute metadata create dict {name: index}
    finaldict = {}
    for i,d in enumerate(attribute_metadata):
        # should make it so that {winner : 0} etc
        #print d.values()[0]

        finaldict[d.values()[1]] = i

    listOfKeys = finaldict.keys()

    maxGainRatio = 0
    maxAttribute = False
    maxAttrSplitValue = [0]
    for e in range(len(finalArray)):

        temp = finalArray[e]

        attributeName = temp[0]
        print attributeName
        if finaldict[attributeName] == 0:
            print "SKIPPING"
            continue
        # gainDict is dict where {gainRatio: False/splitvalue}
        gainThreshold = temp[1]
        # gain_ratio is value of gain ratio
        gain_ratio = gainThreshold[0]

        if gain_ratio >= maxGainRatio:
            maxGainRatio = gain_ratio
            maxAttribute = attributeName
            # this will either be False of the splitValue

            maxAttrSplitValue[0] = gainThreshold[1]




    if maxAttribute == False:
        print "false"
        print data_set

    for key in listOfKeys:
        if maxAttribute == key:
            print "Max:", maxAttribute
            maxAttribute = finaldict[key]




    if maxAttrSplitValue[0] == 0:
       del maxAttrSplitValue[0]
       maxAttrSplitValue.append(False)

    if numerical_splits_count[maxAttribute] == 0:
        return False

    #return (maxAttribute, maxAttrSplitValue[0])
    return (maxAttribute, maxAttrSplitValue[0])

    # best attribute is calculated using gain_ratio_nominal or gain_ratio_numeric

    # if attribute is numerical send split count through


# # ======== Test Cases =============================
# numerical_splits_count = [20,20]

# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)

# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''

    # Find freq of each element
    max_e = [None, 0]
    freq = {}
    for e in data_set:
        if e[0] not in freq.keys():
            freq[e[0]] = 1
        else:
            freq[e[0]] += 1

        if freq[e[0]] > max_e[1]:
            max_e = [e[0], freq[e[0]]]

    return max_e[0]
# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''

    # Get frequencies of each result
    freq = {}
    for e in data_set:
        if e[0] not in freq.keys():
            freq[e[0]] = 1
        else:
            freq[e[0]] += 1

    # Calculate entropy using sum
    entropy = 0
    total = len(data_set)
    for key, value in freq.iteritems():
        p = float(value)/total
        entropy += p * math.log(p, 2)
    entropy *= -1
    return entropy


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    freq = {}
    # Looks like {
    #     attr1: {
    #         output1: freq,
    #         output2: freq
    #     },
    #     attr2: {
    #         output1: freq,
    #         output2: freq
    #     }
    # }

    for e in data_set:
        if e[attribute] not in freq.keys():
            freq[e[attribute]] = {}

        if e[0] not in freq[e[attribute]].keys():
            freq[e[attribute]][e[0]] = 1
        else:
            freq[e[attribute]][e[0]] += 1

    # Calculate gain from sum
    gain = 0
    T = len(data_set)
    for attr_val, attr_freq in freq.iteritems():
        # Calculate T_i. T = len(data_set).
        T_i = 0
        for key, value in attr_freq.iteritems():
            T_i += value

        # Calculate H(X,T)
        H = 0
        for key, value in attr_freq.iteritems():
            p = float(value)/T_i
            H -=  p * math.log(p, 2)

        gain -= float(T_i)/T * H

    gain += entropy(data_set)
    split_info = entropy(map(lambda x: [x[1]], data_set))
    if split_info == 0:
        return 0

    gain_ratio = gain/split_info

    return gain_ratio
# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151

# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741

# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    # Discretize data_set
    data_set2 = sorted(data_set, key=lambda x: x[attribute])
    min_val = data_set2[0][attribute]

    best_split = {
        'threshold': None,
        'gain_ratio': 0
    }
    for i, threshold in enumerate([e[attribute] for e in data_set]):
        if threshold == min_val:
            continue
        if i%steps != 0:
            continue

        discrete_data = []
        for e in data_set:
            e2 = deepcopy(e)
            if e[attribute] >= threshold:
                e2[attribute] = 0
            else:
                e2[attribute] = 1
            discrete_data.append(e2)

        gain_ratio = gain_ratio_nominal(discrete_data, attribute)
        if gain_ratio > best_split['gain_ratio']:
            best_split['threshold'] = threshold
            best_split['gain_ratio'] = gain_ratio

    return (best_split['gain_ratio'], best_split['threshold'])


# ======== Test case =============================
# data_set,attr,step = [[0,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)

# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)

# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    split = {}
    for e in data_set:
        if e[attribute] not in split.keys():
            split[e[attribute]] = []

        split[e[attribute]].append(e)

    return split

# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    split = ([], [])
    for e in data_set:
        if e[attribute] >= splitting_value:
            split[1].append(e)
        else:
            split[0].append(e)

    return split
# ======== Test case =============================
# d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
# split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
# d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
# split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])
