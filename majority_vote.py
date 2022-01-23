import sys

def train_majorityvote(train_data):
    outcomes = dict()
    last_index = len(train_data[0])-1
    for element in train_data:
        if element[last_index] in outcomes:
            outcomes[element[last_index]] += 1
        else:
            outcomes[element[last_index]] = 1
    frequency = 0
    label = 'Z'
    for outcome in outcomes:
        if outcomes[outcome] > frequency:
            if outcomes[outcome] == frequency:
                if outcome < label:
                    frequency = outcomes[outcome]
                    label = outcome
            else:
                frequency = outcomes[outcome]
                label = outcome
    return label

def get_traindata(input_file):
    with open(input_file) as f:
        train_input = f.readlines()
    train_data = []
    for i in range(1,len(train_input)):
        element = train_input[i]
        clean_row = element.split('\t')
        last_index = len(clean_row)-1
        ### get rid of \n
        clean_row[last_index] = clean_row[last_index][:len(clean_row[last_index])-1]
        train_data.append(clean_row)
    return train_data

def file_out_labels(model,data,output_file):
    f = open(output_file,'w')
    for element in data:
        f.write(model)
        f.write('\n')
    f.close()

def get_metrics(model,data):
    total = 0
    mistakes = 0
    last_index = len(data[0]) - 1
    for element in data:
        total += 1
        if element[last_index] == model:
            mistakes += 1
    return str(mistakes/total)

def file_out_metrics(error_train,error_test,metrics_file):
    f = open(metrics_file,'w')
    f.write("error(train): ")
    f.write(error_train)
    f.write('\n')
    f.write("error(test): ")
    f.write(error_test)
    f.close()

def main():
    train_data = get_traindata(sys.argv[1])
    model = train_majorityvote(train_data)
    train_output_file = sys.argv[3]
    file_out_labels(model,train_data,train_output_file)
    test_data = get_traindata(sys.argv[2])
    test_output_file = sys.argv[4]
    file_out_labels(model,test_data,test_output_file)
    metrics_file = sys.argv[5]
    error_train = get_metrics(model,train_data)
    error_test = get_metrics(model,test_data)
    file_out_metrics(error_train, error_test, metrics_file)

main()
