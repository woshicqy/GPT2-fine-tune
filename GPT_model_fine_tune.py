from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline,GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse

import re
import pandas as pd
import datetime
import torch

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from config import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    


def load_dataset(file_path, tokenizer, block_size = 4):
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


def train(args,date):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    if args.model_name == 'gpt2':
        ### GPT-2 fine tune ###
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
    elif args.model_name == 'chatglm-6b':
    ### Ghatglm fine tune ###
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,trust_remote_code=True)
        if torch.cuda.is_available():
            print('>>>>>> Using %s to train your model <<<<<<'%device)
            
            model = AutoModel.from_pretrained(args.model_name,trust_remote_code=True).cuda()
        else:

            model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).float()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=args.inference_mode,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules
    )

    model = get_peft_model(model, peft_config)
    torch.cuda.empty_cache()

    train_dataset = load_dataset(args.train_file_path, tokenizer,args.batch_size)
    data_collator = load_data_collator(tokenizer)
    # tokenizer.save_pretrained(output_dir)
    # model.save_pretrained(output_dir)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        optim=args.optim,

        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.lr,
        save_steps=args.save_steps,
        save_total_limit = args.save_total_limit,
        fp16=args.fp16,
        tf32=args.tf32,

        push_to_hub=args.push2hub,
        remove_unused_columns=args.remove_unused_columns,
        ignore_data_skip=args.ignore_data_skip,
        dataloader_pin_memory=args.dataloader_pin_memory
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        )
    # model_path = args.output_dir+'_'+str(args.save_steps)+'_'+str(args.num_train_epochs)+'_'+str(date)
    model_path = args.output_dir
    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    # print('>>>>> New model is saved! <<<<<')


if __name__ == '__main__':

    config = ini_config()

    import os
    today = datetime.date.today()
    # train_file_path = "D:\cqy\Repos\gpt-fine-tune\Chinese-corpus-collection\训练测试数据\Chinese-qa-content.txt"

    # train_file_path = "D:\cqy\Repos\gpt-fine-tune\Chinese-corpus-collection\训练测试数据\Chinese-qa-without-content.txt"
    # model_name = 'chatglm-6b'
    
    path = os.getcwd()
    # print(f'path:{path}')
    output_dir = config.output_dir
    new_output_dir = output_dir+str(today)
    path = os.path.join(path,new_output_dir)
    # print(f'path:{path}')
    isExist = os.path.exists(path)
    # print(isExist)
    # exit()
    if not isExist:
        os.makedirs(path)
        print('>>>>>> Model Folder is created <<<<<<')

    os.environ["WANDB_DISABLED"] = str(config.WANDB_DISABLED).lower()

    
    train(config, date=today)