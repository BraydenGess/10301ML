import sys
import numpy
from utils import *

def get_parameters(sequences,tags,index_to_word,index_to_tag):
    classification_count = len(index_to_tag)
    feature_count = len(index_to_word)
    hmm_init = np.zeros(classification_count)
    hmm_trans = np.zeros((classification_count,classification_count))
    hmm_emit = np.zeros((classification_count,feature_count))
    rows = len(sequences)
    for i in range(rows):
        for j in range(len(sequences[i])):
            word = index_to_word[sequences[i][j]]
            tag = index_to_tag[tags[i][j]]
            if (j==0):
                hmm_init[tag] += 1
            else:
                prev_tag = index_to_tag[tags[i][j-1]]
                hmm_trans[prev_tag][tag] += 1
            hmm_emit[tag][word] += 1
    hmm_init += 1
    hmm_init = hmm_init/np.sum(hmm_init)
    hmm_trans += 1
    for i in range(len(hmm_trans)):
        denominator = sum(hmm_trans[i])
        numerator = hmm_trans[i]
        hmm_trans[i] = numerator/denominator
    hmm_emit += 1
    hmm_emit_sum = np.transpose(np.asmatrix(np.sum(hmm_emit,axis=1)))
    hmm_emit = np.asmatrix(hmm_emit/hmm_emit_sum)
    return hmm_init,hmm_trans,hmm_emit

def write_hmm_init(hmm_init,file):
    np.savetxt(file,hmm_init,fmt='%.6f')

def write_hmm_trans(hmm_trans,file):
    np.savetxt(file,hmm_trans,fmt='%.6f')

def write_hmm_emit(hmm_emit,file):
    np.savetxt(file,hmm_trans, fmt='%.6f')

def main():
    train_file = sys.argv[1]
    index_to_word_file = sys.argv[2]
    index_to_tag_file = sys.argv[3]
    hmminit_file = sys.argv[4]
    hmmemit_file = sys.argv[5]
    hmmtrans_file = sys.argv[6]
    sequences,tags = parse_file(train_file)
    index_to_word = make_dict(index_to_word_file)
    index_to_tag = make_dict(index_to_tag_file)
    hmm_init,hmm_trans,hmm_emit = get_parameters(sequences,tags,index_to_word,index_to_tag)
    write_hmm_init(hmm_init,hmminit_file)
    write_hmm_trans(hmm_trans,hmmtrans_file)
    write_hmm_trans(hmm_emit,hmmemit_file)


main()
