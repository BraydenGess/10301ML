import sys
import numpy as np
from numpy import genfromtxt
import math

def numpy_matrix(file):
    data = genfromtxt(file, delimiter='\t')
    return data

def important_values(labels,attributes,theta,rows,cols):
    y_hat = np.matmul(attributes, theta)
    dot_product = np.dot(labels,np.transpose(y_hat))
    y0x = dot_product[0][0]
    e0x = np.power(math.e,y_hat)
    plus_one = np.ones([rows,1])
    one_plus_e0x = np.add(plus_one,e0x)
    log = np.log(one_plus_e0x)
    return y_hat,e0x,one_plus_e0x,log,y0x

def J0(labels,attributes,theta,rows,cols):
    y_hat,e0x,one_plus_e0x,log,y0x = important_values(labels,attributes,theta,rows,cols)
    OBJ_F = (1/rows)*(y0x + np.sum(log))
    return OBJ_F,y_hat,e0x,one_plus_e0x,log,y0x

def derivative_J(labels,attributes,theta,learning_rate,rows,cols,y_hat,e0x,one_plus_e0x):
    inverse_one_plus_e0x = np.power(one_plus_e0x,-1)
    right_part = np.matmul(inverse_one_plus_e0x,np.transpose(e0x))
    new_right_part = np.diagonal(right_part)
    new_right_part = np.diagonal(np.subtract(labels,new_right_part))
    gradient = np.matmul(np.transpose(attributes),new_right_part)
    change = learning_rate*gradient
    change.resize((cols,1))
    new_theta = np.add(theta,change)
    return new_theta

def one_iteration(labels,attributes,learning_rate,theta,rows,cols):
    OBJ_F,y_hat,e0x,one_plus_e0x,log,y0x = J0(labels, attributes, theta, rows, cols)
    new_theta = derivative_J(labels,attributes,theta,learning_rate,rows,cols,y_hat,e0x,one_plus_e0x)
    return new_theta,OBJ_F

def split_data(matrix):
    (rows, cols) = matrix.shape
    labels = matrix[:, :1]
    attributes = matrix[:, 1:]
    attribute_bias = np.ones([1, rows])
    attributes = np.insert(attributes, 0, attribute_bias, axis=1)
    return labels,attributes

def train_model(matrix,num_epochs,learning_rate):
    (rows,cols) = matrix.shape
    theta = np.zeros([cols,1])
    labels, attributes = split_data(matrix)
    count = 1
    for i in range(num_epochs):
        theta,OBJ_F = one_iteration(labels,attributes,learning_rate,theta,rows,cols)
    return theta,labels,attributes

def test_model(weights,attributes,labels,file):
    [rows,cols] = labels.shape
    answers = np.matmul(np.transpose(weights),np.transpose(attributes))
    errors = 0
    with open(file,'w') as f:
        f.close()
    for i in range(rows):
        pred = 1/(1+math.e**(-1*answers[0][i]))
        actual = labels[i]
        if pred >= .5:
            pred = 1
        else:
            pred = 0
        if (pred != actual):
            errors += 1
        with open(file,'a') as f:
            f.write(str(pred))
            if (i != rows-1):
                f.write('\n')
    return (errors/rows)

def metrics_out(file,train_error,test_error):
    with open(file,'w') as f:
        f.write('error(train): ')
        f.write(str(train_error))
        f.write('\n')
        f.write('error(test): ')
        f.write(str(test_error))

def main():
    #train_input = 'feature_output.txt'
    #validation_input = 'format_valid.txt'
    #test_input = 'format_test.txt'
    #train_out_file = 'train_labels.txt'
    #test_out_file = 'test_label.txt'
    #metrics_file = 'metrics.txt'
    #num_epochs = 500
    #learning_rate = .00001
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    train_out_file = sys.argv[4]
    test_out_file = sys.argv[5]
    metrics_file = sys.argv[6]
    num_epochs = int(sys.argv[7])
    learning_rate = float(sys.argv[8])

    ###train the model
    training_data = numpy_matrix(train_input)
    weights,training_labels,attributes = train_model(training_data,num_epochs,learning_rate)
    ###testing the model###
    train_error = test_model(weights,attributes,training_labels,train_out_file)
    testing_data = numpy_matrix(test_input)
    test_labels, test_attributes = split_data(testing_data)
    test_error = test_model(weights,test_attributes,test_labels,test_out_file)
    metrics_out(metrics_file,train_error,test_error)



if __name__ == "__main__":
    main()
