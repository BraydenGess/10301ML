import math
import sys
import numpy as np

def numpy_matrix(file):
    data = np.loadtxt(file, delimiter=',')
    return data

def split_data(matrix):
    (rows, cols) = matrix.shape
    labels = matrix[:,:1]
    attributes = matrix[:,1:]
    attribute_bias = np.ones([1, rows])
    attributes = np.insert(attributes, 0, attribute_bias, axis=1)
    return labels,attributes

def add_bias_term(matrix):
    (rows, cols) = matrix.shape
    attribute_bias = np.ones([1, rows])
    attributes = np.insert(matrix, 0, attribute_bias, axis=1)
    return attributes

def add_bias_term_zero(matrix):
    (rows, cols) = matrix.shape
    attribute_bias = np.zeros([1, rows])
    attributes = np.insert(matrix, 0, attribute_bias, axis=1)
    return attributes

def sigmoid(x):
    result = 1 / (1 + math.e ** (-1 * x))
    return result

def softmax(x):
    summation = np.sum(math.e**x)
    result = (math.e**x)/summation
    return result

def loss_function(Y,y_hat):
    (rows,cols) = y_hat.shape
    labels = np.zeros((1,cols))
    labels[0][int(Y)] = 1
    new_yhat = np.log(y_hat)
    result = -1*np.matmul(labels,np.transpose(new_yhat))
    return result

def FeedForward(X,Y,alpha,beta):
    raw_activation = np.matmul(X,np.transpose(alpha))
    z_old = sigmoid(np.asarray(raw_activation))
    z = add_bias_term(z_old)
    raw_output = np.asarray(np.matmul(z, np.transpose(beta)))
    output = softmax(np.asarray(raw_output))
    hot_vector = np.zeros(output.shape[1])
    hot_vector[int(Y)] = 1
    return output,hot_vector,z,z_old

def sigmoid_inverse(matrix):
    (rows,cols) = matrix.shape
    one_array = np.ones((rows,cols))
    second_array = np.power(math.e,-1*matrix)
    final_array = second_array+one_array
    return np.multiply(final_array,matrix)

def FeedBackward(X,output,hot_vector,alpha,beta,z,z_old):
    dldb = output - hot_vector
    dldB = np.matmul(np.transpose(dldb), z)
    real_beta = beta[:,1:]
    dldz = np.matmul(dldb, real_beta)
    one = np.multiply(z_old, dldz)
    two = np.multiply(one, (1 - z_old))
    dalpha = np.matmul(np.transpose(two), X)
    return dldB,dalpha

def shuffle(X, y, epoch):
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def cross_entropy(attributes,labels,alpha,beta):
    (label_rows,label_cols) = labels.shape
    raw_activation = np.matmul(attributes, np.transpose(alpha))
    z_old = sigmoid(np.asarray(raw_activation))
    z = add_bias_term(z_old)
    raw_output = np.asarray(np.matmul(z, np.transpose(beta)))
    set_up = raw_output - np.max(raw_output, axis=1)[:, np.newaxis]
    output = np.exp(set_up) / np.sum(np.exp(set_up), axis=1)[:, np.newaxis]
    (output_rows,output_cols) = output.shape
    hot_matrix = np.zeros((label_rows,output_cols))
    for i in range(label_rows):
        hot_matrix[i][int(labels[i])] = 1
    new_hotmatrix = np.transpose(np.asmatrix(hot_matrix))
    output = np.log(output)
    x = np.diagonal(np.matmul(output,new_hotmatrix))
    x = np.sum(x)
    return (x/label_rows)

def adjust_weights(S_beta, S_alpha, alpha, beta, dldB, dalpha, learning_rate):
    ###Beta
    S_beta_one = S_beta + np.power(dldB, 2)
    beta_lower = S_beta_one + .00001
    beta_lower = np.power(beta_lower, .5)
    beta_upper = learning_rate * dldB
    beta_finish = beta_upper / beta_lower
    beta = beta - beta_finish
    ###Alpha
    S_alpha_one = S_alpha + np.power(dalpha, 2)
    alpha_lower = S_alpha_one + .00001
    alpha_lower = np.power(alpha_lower, .5)
    alpha_upper = learning_rate * dalpha
    alpha_finish = alpha_upper / alpha_lower
    alpha = alpha - alpha_finish
    return alpha, beta, S_alpha_one, S_beta_one

def write_metrics_entropy(train_entropy,valid_entropy,epoch,file):
    with open(file,'a') as f:
        f.write(f'epoch={epoch+1} crossentropy(train): {train_entropy}')
        f.write('\n')
        f.write(f'epoch={epoch+1} crossentropy(validation): {valid_entropy}')
        f.write('\n')
        f.close()

def write_metrics(train_error,valid_error,file):
    with open(file,'a') as f:
        f.write(f'error(train): {train_error}')
        f.write('\n')
        f.write(f'error(validation): {valid_error}')
        f.close()

def train_neuralnetwork(attributes,labels,hidden_units,init_flag,num_epochs,learning_rate,valid_attributes,valid_labels,file):
    (rows,cols) = attributes.shape
    S_beta = np.zeros((10,hidden_units+1))
    S_alpha = np.zeros((hidden_units,cols))
    if init_flag == 2:
        alpha = np.zeros((hidden_units,cols))
        beta = np.zeros((10,hidden_units+1))
    if init_flag == 1:
        alpha = np.random.uniform(low=-0.1, high=0.1, size=(hidden_units,cols-1))
        beta = np.random.uniform(low=-0.1, high=0.1, size=(10,hidden_units))
        alpha = add_bias_term_zero(alpha)
        beta = add_bias_term_zero(beta)
    for i in range(num_epochs):
        new_attributes,new_labels = shuffle(attributes,labels,i)
        for j in range(rows):
            X = np.asmatrix(new_attributes[j])
            Y = new_labels[j]
            output,hot_vector,z,z_old = FeedForward(X,Y,alpha,beta)
            dldB,dalpha = FeedBackward(X,output,hot_vector,alpha,beta,z,z_old)
            alpha,beta,S_alpha,S_beta = adjust_weights(S_beta, S_alpha, alpha, beta, dldB, dalpha,learning_rate)
            alpha = alpha[:,1:]
            alpha = add_bias_term_zero(alpha)
        train_entropy = -1*cross_entropy(attributes, labels, alpha, beta)
        validation_entropy = -1*cross_entropy(valid_attributes,valid_labels, alpha, beta)
        write_metrics_entropy(train_entropy, validation_entropy,i, file)
    return alpha,beta

def predict(attributes,labels,alpha,beta,file):
    with open(file,'w') as f:
        (rows, cols) = attributes.shape
        correct = 0
        for j in range(rows):
            X = attributes[j]
            Y = labels[j]
            output,hot_vector,z,z_old = FeedForward(X,Y,alpha,beta)
            prediction = int(np.argmax(output))
            actual = int(Y[0])
            if actual == prediction:
                correct += 1
            f.write(str(prediction))
            f.write('\n')
    return (1 - (correct/rows))


def main():
    #train_input = '/Users/BradyGess/Downloads/hw5/handout/small_train_data.csv'
    #valid_input = '/Users/BradyGess/Downloads/hw5/handout/small_validation_data.csv'
    #train_out = 'train_labels.txt'
    #validation_out = 'output.txt'
    #metrics_out = 'metrics.txt'
    #num_epochs = 500
    #hidden_units = 4
    #init_flag = 2
    #learning_rate = 0.1
    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])
    with open(metrics_out,'w') as f:
        f.close()
    train_data = numpy_matrix(train_input)
    train_labels,train_attributes = split_data(train_data)
    train_attributes = add_bias_term(train_attributes)
    valid_data = numpy_matrix(valid_input)
    valid_labels, valid_attributes = split_data(valid_data)
    valid_attributes = add_bias_term(valid_attributes)
    alpha,beta = train_neuralnetwork(train_attributes,train_labels,hidden_units,init_flag,num_epochs,learning_rate,
                                     valid_attributes,valid_labels,metrics_out)
    train_error = predict(train_attributes,train_labels,alpha,beta,train_out)
    validation_error = predict(valid_attributes,valid_labels, alpha, beta,validation_out)
    write_metrics(train_error,validation_error, metrics_out)

main()
