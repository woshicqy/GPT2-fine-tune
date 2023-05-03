import pandas as pd
import numpy as np
import json
import os


import argparse
import collections
def load_txt(filename):
    data = np.genfromtxt(filename,dtype=str,encoding='utf-8')
    # print('data:',data)
    return data


def build_hashmap(path):
    all_paths = collections.defaultdict(list)
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            all_paths[parent].append(filename)
    return all_paths


def save_file(path,data):        
    np.savetxt(path,data,fmt='%s',newline='\n',encoding='utf-8')


def main(args):
    all_paths = build_hashmap(args.corpus_path)
    all_paths_keys = all_paths.keys()
    empty_files = []
   
    for key in all_paths_keys:
        file_list = all_paths[key]
        main_content = []
        appendix_content = []
        for tmp in file_list:
            idx = tmp.split(' ')[0]
            # print(idx)
            try:
                float(idx)
                main_content.append(tmp)
            except:
                appendix_content.append(tmp)


        sorted_paths = sorted(main_content,key=lambda x: float(x.split(' ')[0] ))
        preprocessed_content = sorted_paths + appendix_content

        all_content = []
        
        for filename in preprocessed_content:
            file_path = os.path.join(key,filename)
            sz = os.path.getsize(file_path)
            if not sz:
                # print('filename:',file_path)
                empty_files.append(file_path)
                continue
            else:
                # print('filename:',file_path)
                cur_content = load_txt(file_path)
            # print((cur_content).size)
            if (cur_content).size == 0:
                empty_files.append(file_path)
                # print('filename:',file_path)
                continue
            elif cur_content not in all_content:
                all_content.append(cur_content)
            else:
                continue
        
        path = os.getcwd()
        # print(f'path:{path}')
        output_dir = args.out_path
        path = os.path.join(path,output_dir)
        # print(f'path:{path}')
        # exit()
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)
            print('>>>>>> Model Folder is created <<<<<<')
        sep = os.sep
        key = key.split(sep)[-1] + '.txt'
        # print(f'key:{key}')
        out_path = os.path.join(path,key) 
        log_out_path = os.path.join(path,'log')
        log_isExist = os.path.exists(log_out_path)
        if not log_isExist:
            os.makedirs(log_out_path)

        save_file(out_path,all_content)
    # print(empty_files)
    # exit()
    log_out_path = log_out_path + '\log.txt'
    save_file(log_out_path,empty_files)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # basic info
    parser.add_argument('--corpus_path',           default = 'E:\cqy-gpt\gpt-fine-tune\Chinese-corpus-collection\训练测试数据\规范',               type = str,
                        help = 'your data saving path')
    
    parser.add_argument('--out_path',              default = 'E:\cqy-gpt\gpt-fine-tune\Chinese-corpus-collection\训练测试数据\规范merged',         type = str,
                        help = 'your merged data saving path')
    
    args = parser.parse_args()

    path = args.corpus_path
    main(args)
    # filename = os.path.join("E:\\cqy-gpt\\gpt-fine-tune\\Chinese-corpus-collection\\训练测试数据\\混凝土外加剂应用技术规范[附条文说明]",'1 总 则.txt')
    # test_file = "E:\cqy-gpt\gpt-fine-tune\Chinese-corpus-collection\训练测试数据\规范\地铁设计规范[附条文说明]\附录E 缓和曲线地段矩形隧道建筑限界加宽计算.txt"
    # test_re = load_txt(test_file)
    # print(f'test_re:{test_re}')
    print('>>>>> Files are merged <<<<<')