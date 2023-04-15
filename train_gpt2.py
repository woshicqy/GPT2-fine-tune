from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline,GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments




# tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")


# text_generator = TextGenerationPipeline(model, tokenizer)   
# output = text_generator("这是很久之前的事情了", max_length=100, do_sample=True)
# print(f'output:{output}')

import re
import pandas as pd
import datetime
import torch
def cleaning(s):
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

def cvs2txt(origin_filename,saving_file):

    df = pd.read_csv(origin_filename, encoding="ISO-8859-1") 
    df = df.dropna()
    text_data = open(saving_file, 'w')
    for idx, item in df.iterrows():
        article = cleaning(item["Article"])
        text_data.write(article)
    text_data.close()
    print('txt data is saved!')



def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator


def train(train_file_path,model_name,output_dir,overwrite_output_dir,per_device_train_batch_size,num_train_epochs,save_steps,date):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('>>>>>> Using %s to train your model <<<<<<'%device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)
    # tokenizer.save_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # model.save_pretrained(output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        optim='adamw_torch',
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        )
    model_path = output_dir+'_'+str(save_steps)+'_'+str(num_train_epochs)+'_'+str(date)
    trainer.train()
    trainer.save_model(model_path)
    print('>>>>> New model is saved! <<<<<')


if __name__ == '__main__':
    import os
    origin_filename = "Articles.csv"
    saving_file = "Articles.txt"
    cvs2txt(origin_filename,saving_file)


    today = datetime.date.today()
    train_file_path = "Articles.txt"
    model_name = 'gpt2'
    
    path = os.getcwd()
    # print(f'path:{path}')
    output_dir = 'gpt2-fine-tune/'
    new_output_dir = output_dir+str(today)
    path = os.path.join(path,new_output_dir)
    # print(f'path:{path}')
    isExist = os.path.exists(path)
    # print(isExist)
    # exit()
    if not isExist:
        os.makedirs(path)
        print('>>>>>> Model Folder is created <<<<<<')

    os.environ["WANDB_DISABLED"] = "true"
    overwrite_output_dir = False
    per_device_train_batch_size = 8
    num_train_epochs = 5.0
    save_steps = 500

    train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    date=today)