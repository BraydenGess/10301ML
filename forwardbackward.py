import sys
import numpy
from utils import *

def reverse_dict(a):
    b = dict()
    for element in a:
        b[a[element]] = element
    return b

def compute_prevsum(alpha,i,B,j):
    summation = 0
    (rows,cols) = alpha.shape
    for k in range(cols):
        summation += (alpha[i-1][k]*B[k][j])
    return summation

def forward(sentence,pi,A,B,index_to_words,index_to_tag):
    alpha = np.zeros((len(sentence),len(index_to_tag)))
    for i in range(len(sentence)):
        for j in range(len(index_to_tag)):
            if i==0:
                alpha[i][j] = pi[j]*A[j][index_to_words[sentence[i]]]
            else:
                alpha[i][j] = A[j][index_to_words[sentence[i]]]*compute_prevsum(alpha,i,B,j)
    return alpha

def compute_prevsum_backward(beta,A,B,i,j,index_to_words,key):
    summation = 0
    (rows, cols) = beta.shape
    for k in range(cols):
        summation += (A[k][key]*beta[i+1][k]*B[j][k])
    return summation

def backward(sentence,pa,A,B,index_to_words,index_to_tag):
    beta = np.zeros((len(sentence),len(index_to_tag)))
    for i in range(len(sentence)-1,-1,-1):
        for j in range(len(index_to_tag)):
            if i == len(sentence)-1:
                beta[i][j] = 1
            else:
                key = index_to_words[sentence[i+1]]
                beta[i][j] = compute_prevsum_backward(beta,A,B,i,j,index_to_words,key)
    return beta


def forwardbackward(sequences,tags,pi,A,B,index_to_word,index_to_tag,predicted_file,tag_to_index):
    with open(predicted_file,'w') as f:
        correct = 0
        total = 0
        log_sum = 0
        for i in range(0,len(sequences)):
            alpha = forward(sequences[i],pi,A,B,index_to_word,index_to_tag)
            beta = backward(sequences[i],pi,A,B,index_to_word,index_to_tag)
            prob = alpha*beta
            log_sum += np.log(np.sum(alpha[len(alpha)-1]))
            tags_index = np.argmax(prob,axis=1)
            for guess in range(len(tags_index)):
                total += 1
                if index_to_tag[tags[i][guess]] == tags_index[guess]:
                    correct += 1
                word = sequences[i][guess]
                tag = tag_to_index[tags_index[guess]]
                f.write(str(word))
                f.write('\t')
                f.write(str(tag))
                f.write('\n')
            f.write('\n')
        accuracy = correct/total
        log_liklihood = log_sum/len(sequences)
    f.close()
    return accuracy,log_liklihood



def main():
    validation_input_file = sys.argv[1]
    index_to_word_file = sys.argv[2]
    index_to_tag_file = sys.argv[3]
    hmminit_file = sys.argv[4]
    hmmemit_file = sys.argv[5]
    hmmtrans_file = sys.argv[6]
    predicted_file = sys.argv[7]
    metrics_file = sys.argv[8]
    sequences, tags = parse_file(validation_input_file)
    index_to_word = make_dict(index_to_word_file)
    index_to_tag = make_dict(index_to_tag_file)
    tag_to_index = reverse_dict(index_to_tag)
    pi = np.genfromtxt(hmminit_file,dtype=np.float64)
    A = np.genfromtxt(hmmemit_file, dtype=np.float64)
    B = np.genfromtxt(hmmtrans_file, dtype=np.float64)
    accuracy,log_liklihood = forwardbackward(sequences,tags,pi,A,B,index_to_word,index_to_tag,predicted_file,tag_to_index)
    write_metrics(metrics_file,log_liklihood,accuracy)




main()
