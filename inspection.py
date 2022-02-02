import sys
import csv
import math

def get_data(file):
    data = []
    with open(file) as f:
        tsv_file = csv.reader(f,delimiter="\t")
        for line in tsv_file:
            data.append(line)
    return data

def calculate_entropy(data):
    outcomes_dict = {}
    sum = len(data)-1
    for i in range(1,len(data)):
        outcome = data[i][len(data[i])-1]
        if outcome in outcomes_dict:
            outcomes_dict[outcome] += 1
        else:
            outcomes_dict[outcome] = 1
    entropy = 0
    for element in outcomes_dict:
        fraction = (outcomes_dict[element]/sum)
        entropy -= (fraction * math.log2(fraction))
    return entropy,outcomes_dict

def calculate_error(outcome_dict):
    sum = 0
    min = None
    first = True
    for element in outcome_dict:
        sum += outcome_dict[element]
        if first == True:
            min = outcome_dict[element]
        else:
            if outcome_dict[element]<min:
                min = outcome_dict[element]
    return min/sum

def write_to_file(entropy,error,file):
    with open(file,'w') as f:
        f.write("entropy: ")
        f.write(str(entropy))
        f.write('\n')
        f.write("error: ")
        f.write(str(error))

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data = get_data(input_file)
    entropy,outcome_dict = calculate_entropy(data)
    error = calculate_error(outcome_dict)
    write_to_file(entropy, error,output_file)


if __name__ == "main":
    main()
