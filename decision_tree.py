from inspection import *
import sys

class Node:
    def __init__(self,attribute=None,left=None,right=None,left_label=None,right_label=None,dict=None,value=None):
        self.attribute = attribute
        self.left = left
        self.right = right
        self.left_label = left_label
        self.right_label = right_label
        self.dict = dict
        self.value = value

###Helper Functions
def get_outcomes_list(dict):
    outcome_list = []
    for element in dict:
        outcome_list.append(element)
    return outcome_list

def sort_dictionary(dict):
    list = []
    new_dict = {}
    for element in dict:
        list.append(element)
    list = sorted(list)
    for element in list:
        new_dict[element] = dict[element]
    return new_dict

def dict_to_output(dict):
    text = '['
    for element in dict:
        text += str(dict[element]) + ' '
        text += str(element)
        text += '/'
    text = text[:len(text)-1]
    text += ']'
    return text
####

def check_pure(data,outcome_list):
    outcome_dict = {}
    for element in outcome_list:
        outcome_dict[element] = 0
    for i in range(1,len(data)):
        outcome = data[i][len(data[i])-1]
        outcome_dict[outcome] += 1
    for element in outcome_dict:
        if outcome_dict[element] == 0:
            return outcome_dict
    return False

def majority_vote(data,outcome_list):
    outcome_dict = {}
    for element in outcome_list:
        outcome_dict[element] = 0
    for i in range(1, len(data)):
        outcome = data[i][len(data[i]) - 1]
        outcome_dict[outcome] += 1
    max = -1
    max_label = None
    for element in outcome_dict:
        if outcome_dict[element] > max:
            max = outcome_dict[element]
            max_label = element
        elif outcome_dict[element] == max:
            if element<max_label:
                max = outcome_dict[element]
                max_label = element
    return max_label,outcome_dict

def get_mutual_information(data,index,sum,outcome_list):
    attribute_values = {}
    for i in range(1,len(data)):
        attribute = data[i][index]
        response = data[i][len(data[i])-1]
        answer_index = outcome_list.index(response)
        if attribute not in attribute_values:
            attribute_values[attribute] = [0,0,0]
        attribute_values[attribute][answer_index] += 1
        attribute_values[attribute][2] += 1
    attribute_values = sort_dictionary(attribute_values)
    mutual_info = 0
    for element in attribute_values:
        sub_section = 0
        for i in range(2):
            frac = attribute_values[element][i]/attribute_values[element][2]
            if frac != 0:
                sub_section -= (frac*math.log2(frac))
            else:
                sub_section = 0
        sub_section *= (attribute_values[element][2]/sum)
        mutual_info += sub_section
    return mutual_info,attribute_values

def get_best_mutual_information(data,entropy,outcome_list):
    sum = len(data)-1
    max = -1
    max_index = -1
    max_name = -1
    max_attribute_dict = None
    for i in range(len(data[0])-1):
        mutual_info,attribute_dict = get_mutual_information(data,i,sum,outcome_list)
        if entropy-mutual_info > 0:
            if entropy-mutual_info > max:
                max = entropy-mutual_info
                max_index = i
                max_name = data[0][i]
                max_attribute_dict = attribute_dict
    return max_index,max_attribute_dict

def train_tree(data,outcome_list,depth,max_depth,entropy):
    ### Bases Cases
    pure_status = check_pure(data,outcome_list)
    if pure_status != False:
        majority_vote_class, outcome_dict = majority_vote(data, outcome_list)
        new_node = Node(dict=outcome_dict,value=majority_vote_class)
        return new_node
    if depth == max_depth:
        majority_vote_class,outcome_dict = majority_vote(data,outcome_list)
        new_node = Node(dict = outcome_dict,value = majority_vote_class)
        return new_node
    if depth == len(data[0])-1:
        majority_vote_class, outcome_dict = majority_vote(data, outcome_list)
        new_node = Node(dict=outcome_dict, value=majority_vote_class)
        return new_node
    ####
    mutual_index,attribute_dict = get_best_mutual_information(data,entropy,outcome_list)
    label_list = []
    for element in attribute_dict:
        label_list.append(element)
    left_data = []
    right_data = []
    for i in range(len(data)):
        if i == 0:
            left_data.append(data[i])
            right_data.append(data[i])
        else:
            if data[i][mutual_index] == label_list[0]:
                left_data.append(data[i])
            else:
                right_data.append(data[i])
    gc,current_dictionary = majority_vote(data,outcome_list)
    if len(label_list) == 2:
        new_node = Node(attribute=data[0][mutual_index],left_label=label_list[0],right_label=label_list[1],
                    left=train_tree(left_data,outcome_list,depth+1,max_depth,entropy),
                    right=train_tree(right_data,outcome_list,depth+1,max_depth,entropy),
                    dict = current_dictionary)
    else:
        new_node = Node(attribute=data[0][mutual_index], left_label=label_list[0],
                        left=train_tree(left_data, outcome_list, depth + 1, max_depth, entropy),
                        dict=current_dictionary)
    return new_node

def print_tree(root,depth):
    if root.value == None:
        right_text = ''
        left_text = ''
        for i in range(depth):
            right_text += '| '
            left_text += '| '
        if root.right_label != None:
            right_text += root.attribute
            right_text += ' = '
            right_text += root.right_label
            right_text += ': '
            right_text += dict_to_output(root.right.dict)
            print(right_text)
            if root.right.value == None:
                print_tree(root.right,depth+1)
        if root.left_label != None:
            left_text += root.attribute
            left_text += ' = '
            left_text += root.left_label
            left_text += ': '
            left_text += dict_to_output(root.left.dict)
            print(left_text)
            if root.left.value == None:
                print_tree(root.left,depth+1)
    else:
        dict_to_output(root.dict)

def predict_helper(data,root,attributes):
    if root.value == None:
        index = attributes.index(root.attribute)
        value = data[index]
        if value == root.left_label:
            return predict_helper(data,root.left,attributes)
        elif value == root.right_label:
            return predict_helper(data,root.right,attributes)
    else:
        return root.value

def predict(data,root):
    labels = []
    for i in range(1,len(data)):
        labels.append(predict_helper(data[i],root,data[0]))
    return labels

def evaluate(data,predictions):
    total = len(data)-1
    correct = 0
    for i in range(1,len(data)):
        if data[i][len(data[i])-1] == predictions[i-1]:
            correct += 1
    return 1 - correct/total

def writeto_labeling_file(label_file,labels):
    with open(label_file,'w') as f:
        for i in range(len(labels)):
            f.write(labels[i])
            if i != len(labels)-1:
                f.write("\n")

def writeto_metrics(metric_file,training_error,testing_error):
    with open(metric_file,'w') as f:
        f.write("error(train): ")
        f.write(str(training_error))
        f.write("\n")
        f.write("error(test): ")
        f.write(str(testing_error))

def main():
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    max_depth = sys.argv[3]
    train_label_file = sys.argv[4]
    test_label_file = sys.argv[5]
    metric_file = sys.argv[6]
    training_data = get_data(training_file)
    testing_data = get_data(testing_file)
    entropy,outcome_dict = calculate_entropy(training_data)
    outcome_list = get_outcomes_list(outcome_dict)
    outcome_list = sorted(outcome_list)
    root = train_tree(training_data,outcome_list,0,int(max_depth),entropy)
    print(dict_to_output(root.dict))
    print_tree(root,1)
    training_predictions = predict(training_data,root)
    testing_predictions = predict(testing_data, root)
    training_error = evaluate(training_data,training_predictions)
    testing_error = evaluate(testing_data,testing_predictions)
    writeto_labeling_file(train_label_file,training_predictions)
    writeto_labeling_file(test_label_file,testing_predictions)
    writeto_metrics(metric_file, training_error, testing_error)

main()
