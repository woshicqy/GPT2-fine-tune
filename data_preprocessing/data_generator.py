import json
import datetime
import numpy as np
import collections

def loading_data_from_txt(filename):
    origin_data = np.genfromtxt(filename,delimiter='<END>',dtype=str)
    return origin_data

def preprocessing(filename):
    data_from_txt = loading_data_from_txt(filename)
    rows = len(data_from_txt)
    data2json = []
    for i in range(rows):
        content = collections.defaultdict(list)
        content['prompt'] = data_from_txt[i][0] + '->'
        content['completion'] = ' ' + data_from_txt[i][1] + ' END'
        data2json.append(content)

    return data2json

def txt2jsonl(file_txt,file_jsonl):

    data_ready2json_ = preprocessing(file_txt)
    with open(file_jsonl, "w") as output_file:
        for entry in data_ready2json_:
            json.dump(entry, output_file)
            output_file.write("\n")




if __name__ == '__main__':
    today = datetime.date.today()
    savingfile_name = "training_data-" + str(today) + '.jsonl'
    loadingfile_name = 'data_file.txt'
    data_ready2json_ = txt2jsonl(loadingfile_name,savingfile_name)

    print(f'>>>>> Training data generation is done <<<<<')

