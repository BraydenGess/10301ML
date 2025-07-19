import sys
import csv
import numpy as np

def bag_of_words_dict(dict_file):
    new_dict = dict()
    with open(dict_file) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            line = line[0].split(" ")
            new_dict[line[0]] = line[1]
    return new_dict

def clean_reviews(bag_dict,reviews,output_file):
    with open(output_file,'w') as f:
        pass
    count = 0
    length = len(bag_dict)
    with open(reviews,'r') as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            v = [0] * (length+1)
            review = line[1].split(' ')
            v[0] = int(line[0])
            for element in review:
                if element in bag_dict:
                    v[int(bag_dict[element])+1] = 1
            count += 1
            bag_output(v,output_file)

def bag_output(vector,output_file):
    with open(output_file,'a') as f:
        rows = len(vector)
        for j in range(rows):
            if vector[j] >= .5:
                f.write('1')
            else:
                f.write('0')
            if (j != rows-1):
                f.write('\t')
        f.write('\n')

def test_output(format_out,format_test):
    my_data = []
    test_data = []
    with open(format_out) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            my_data.append(line)
    f.close()
    with open(format_test) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            test_data.append(line)
    f.close()
    for i in range(len(my_data)):
        for j in range(len(my_data[i])):
            if abs((float(my_data[i][j])-float(test_data[i][j])))>.00001:
               print(i,j,my_data[i][j],test_data[i][j])

def get_word2vec_dict(dict_file):
    new_dict = dict()
    count = 0
    with open(dict_file) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            new_dict[line[0]] = np.array(line[1:],dtype=np.float64)
    return new_dict

def embed_words(word2vec_dict,input_file,output_file):
    with open(output_file,'w') as f:
        pass
    with open(input_file,'r') as f:
        tsv_file = csv.reader(f, delimiter="\t")
        a = True
        for line in tsv_file:
            result = np.zeros([1,300],dtype=np.float64)
            label = line[0]
            review = line[1].split(" ")
            counter = 0
            for element in review:
                if element in word2vec_dict:
                    counter += 1
                    result = np.add(result,word2vec_dict[element])
            word2_vec_output(result,output_file,counter,label)

def word2_vec_output(vector,output_file,counter,label):
    with open(output_file,'a') as f:
        f.write(label)
        f.write('\t')
        for j in range(300):
            number = round(vector[0,j]/counter,6)
            f.write(str(number))
            if (j != 299):
                f.write('\t')
        f.write('\n')

def main():
    input_file = sys.argv[1]
    validation_file = sys.argv[2]
    test_file = sys.argv[3]
    dict_file = sys.argv[4]
    feature_file = sys.argv[5]
    format_out = sys.argv[6]
    format_validation_out = sys.argv[7]
    format_test_out = sys.argv[8]
    feature_flag = int(sys.argv[9])

    ### Feature Modeling ###
    if feature_flag == 1:
        bag_dict = bag_of_words_dict(dict_file)
        clean_reviews(bag_dict,input_file,format_out)
        clean_reviews(bag_dict,validation_file, format_validation_out)
        clean_reviews(bag_dict,test_file, format_test_out)
    if feature_flag == 2:
        word2vec_dict = get_word2vec_dict(feature_file)
        embed_words(word2vec_dict,input_file,format_out)
        embed_words(word2vec_dict,validation_file, format_validation_out)
        embed_words(word2vec_dict,test_file, format_test_out)

    #test_output(format_out, format_ttrain)

if __name__ == "__main__":
    main()
