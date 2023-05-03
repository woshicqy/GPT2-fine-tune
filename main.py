from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline,GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoConfig
import argparse

import re
import pandas as pd
import datetime
import torch
import os

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import argparse
from config import *



if __name__ == '__main__':
    args = ini_config()

    checkpoint = 'checkpoint-50000' ### choose the saved checkpoint
    saved_model = os.path.join(args.output_dir,checkpoint)

    if args.model_name == 'gpt2':
        ### GPT-2 fine tune ###
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(saved_model)
        text_generator = TextGenerationPipeline(model, tokenizer)
        output = text_generator("江苏路街道", max_length=100, do_sample=True)
        print(f'output:{output}')
        
    elif args.model_name == 'chatglm-6b':
    ### Ghatglm fine tune ###
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, pre_seq_len=128)

        if torch.cuda.is_available():          
            # model = AutoModel.from_pretrained(args.model_name,config=config,trust_remote_code=True).half().cuda()

            model = AutoModel.from_pretrained(args.model_name,config=config,trust_remote_code=True).float()

            # model = AutoModel.from_pretrained(args.model_name)
        else:

            model = AutoModel.from_pretrained(args.model_name,config=config, trust_remote_code=True).float()
        
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

        model.load_state_dict(torch.load(os.path.join(saved_model, "pytorch_model.bin")), strict=False)
        torch.cuda.empty_cache()
        # prefix_state_dict = torch.load(os.path.join(saved_model, "pytorch_model.bin"))
        # new_prefix_state_dict = {}

        # for k, v in prefix_state_dict.items():
        #     new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        # print('prefix:',prefix_state_dict)

        # print('new_prefix_state_dict:',new_prefix_state_dict)
        # print('model:',model)
        # exit()

        # model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        # model = model.half().cuda()
        # model.transformer.prefix_encoder.float()
        
        
        
        model = model.eval()
        response, history = model.chat(tokenizer, "你好", history=[])
        print('Response:',response)
        # response, history = model.chat(tokenizer, "你好", history=[])
        # print('Answer:',response)


    

    
    