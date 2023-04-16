import pandas as pd
import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
import re
import matplotlib
import math


class Analyzer4txt(object):
    """docstring for Analyzer4txt"""
    def __init__(self, args):
        super(Analyzer4txt, self).__init__()
        self.args = args
        self.res = []
        

    def load_data(self):
        df = pd.read_csv(self.args.input_dir, encoding="ISO-8859-1") 
        df = df.dropna()
        return df

    def cleaning(self,s):
        s = str(s)
        s = re.sub('\s\W',' ',s)
        s = re.sub('\W,\s',' ',s)
        s = re.sub("\d+", "", s)
        s = re.sub('\s+',' ',s)
        s = re.sub('[!@#$_]', '', s)
        s = s.replace("co","")
        s = s.replace("https","")
        s = s.replace("[\w*"," ")
        return s
    
    def corpus_nums(self):
        df = self.load_data()
        print(len(df))
        return len(df)

    def sentence_distribution(self):
        df = self.load_data()
        df2numpy = df.to_numpy()

        for i in range(len(df2numpy)):
            tmp = df2numpy[i][0].strip()  ### remove white space, line break at the end of sentence
            tmp_list = tmp.split(' ')
            self.res.append(len(tmp_list))
        return self.res
    
    def report_generate(self):

        self.sentence_distribution()

        max_len = max(self.res)
        min_len = min(self.res)
        avg_len = np.mean(self.res)
        std_len = np.std(self.res)
        content = [['文章最多字数为：', max_len],
                   ['文章最少字数为：', min_len],
                   ['文章平均字数为：', avg_len],
                   ['文章字数方差为：', std_len],      
                   ]
        path = os.getcwd()
        # print(f'path:{path}')
        path = os.path.join(path,self.args.out_dir)
        isExist = os.path.exists(path)
        # print(f'path:{path}')
        if not isExist:
            os.makedirs(path)
            print('>>>>>> Data Report Folder is created <<<<<<')

        out_result = os.path.join(path,self.args.out_filename)
        with open(out_result, 'w') as f:
            for i in range(len(content)):
                if i != (len(content)-1):
                    f.write(content[i][0] + ' ' + str(content[i][1]) +'\n')
                else:
                    f.write(content[i][0] + ' ' + str(content[i][1]))
        f.close()
        print(f'>>>>>> Report saving is done <<<<<<')

    def result_visulization(self):
        self.sentence_distribution()

        max_len = max(self.res)
        min_len = min(self.res)
        avg_len = np.mean(self.res)
        std_len = np.std(self.res)
        
        fig, ax = plt.subplots()
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        # plt.hist(self.res)

        x = range(int(min(self.res) / self.args.interval) * self.args.interval, math.ceil(max(self.res) / self.args.interval) * self.args.interval, self.args.interval)
        # hist = ax.hist(self.res,x,edgecolor="black")
        plt.xlabel('语料所含字数')
        plt.ylabel('频数')
        plt.title('语料字数分布图')
        # counts, edges, bars = plt.hist(self.res)
        # plt.bar_label(bars)

        arr=plt.hist(self.res,bins=self.args.interval)
        for i in range(len(arr[0])):
        #     plt.annotate(text=int(hist[0][i]), xy=(hist[1][i] + self.args.interval / 3, hist[0][i]))
            plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

        path = os.getcwd()
        path = os.path.join(path,self.args.out_dir)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print('>>>>>> Data Report Folder is created <<<<<<')

        out_result = os.path.join(path,self.args.outimg_filename)
        
        plt.show()
        plt.savefig(out_result)
        print(f'>>>>>> figure saving is done <<<<<<')
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic info
    parser.add_argument('--input_dir',              default = 'Articles.csv',       type=str,
                        help = 'your data saving path')
    parser.add_argument('--out_filename',           default = 'result.txt',         type=str,
                        help = 'your analysis result saving file name')
    parser.add_argument('--outimg_filename',        default = 'result_hist.png',           type=str,
                        help = 'your analysis image result saving file name')
    parser.add_argument('--interval',               default = 25,                    type=int,
                        help = 'interval for drawing hist plots')
    parser.add_argument('--out_dir',                default = "data_analysis",      type=str,
                        help = 'your analysis result saving folder path')
    # parse the arguments
    args = parser.parse_args()
    analyzer = Analyzer4txt(args)
    analyzer.report_generate()
    analyzer.result_visulization()
    