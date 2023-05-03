import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ini_config():
    parser = argparse.ArgumentParser()
    # basic info
    parser.add_argument('--train_file_path',              default = "E:\cqy-gpt\gpt-fine-tune\Chinese-corpus-collection\训练测试数据\Chinese-qa-content.txt",         type = str,
                        help = 'your data saving path')

    parser.add_argument('--output_dir',                   default = 'fine_tune_models',                                                                             type = str,
                        help = 'your analysis result saving folder path')
    
    parser.add_argument('--checkpoint',                   default = 'checkpoint-50000',                                                                              type = str,
                        help = 'load your fine tune model')
    

    parser.add_argument('--overwrite_output_dir',         default = False,                                                                                           type = str2bool,
                        help = 'overwrite the output dir')
    
    parser.add_argument('--model_name',                   default = "chatglm-6b",                                                                                    type = str,
                        help = 'your model type')
    
    parser.add_argument("--WANDB_DISABLED",               default = True,                                                                                            type=str2bool, 
                        nargs = '?',  const = True,
                        help ="disable your wandb")
    
    # lora info

    parser.add_argument('--inference_mode',               default = False,                                                                                           type = str2bool,
                        help = 'inference mode')
    parser.add_argument('--lora_r',                       default = 8,                                                                                               type = int,
                        help = 'Lora attention dimension')
    parser.add_argument('--lora_alpha',                   default = 32,                                                                                              type = float,
                        help = 'The alpha parameter for Lora scaling')
    parser.add_argument('--lora_dropout',                 default = 0.1,                                                                                             type = float,
                        help = 'The dropout probability for Lora layers')
    
    parser.add_argument('--target_modules',               default = ['query_key_value'],                                                                             type = str,
                        help = 'target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to')

    
    # hyperparameters
    parser.add_argument('--batch_size',                    default = 16,                                                                                             type = int,
                        help = 'batch size while training your model')
    parser.add_argument('--per_device_train_batch_size',   default = 16,                                                                                             type = int,
                        help = 'batch size while training your model per device')
    parser.add_argument('--num_train_epochs',              default = 5,                                                                                              type = int,
                        help = 'epochs for training')
    
    parser.add_argument('--optim',                         default = 'adamw_torch',                                                                                  type = str,
                        help = 'optimizer used for training')
    
    parser.add_argument('--evaluation_strategy',           default = 'no',                                                                                           type = str,
                        help = '"no": No evaluation is done during training. "steps": Evaluation is done (and logged) every eval_steps. "epoch": Evaluation is done at the end of each epoch.')
    

    parser.add_argument("--fp16",                          default = True,                                                                                           type=str2bool, 
                        nargs = '?',  const = True,
                        help = "using fb16")
    
    parser.add_argument("--tf32",                          default = True,                                                                                           type = str2bool, 
                        nargs = '?',  const = True,
                        help = "using tf32")
    
    parser.add_argument('--gradient_accumulation_steps',   default = 1,                                                                                              type = int,
                        help = 'steps of gradient accumulation')
    
    parser.add_argument('--eval_steps',                    default = 100,                                                                                            type = int,
                        help = 'Number of update steps between two evaluations if evaluation_strategy="steps". Will default to the same value as logging_steps if not set')
    
    parser.add_argument('--logging_steps',                 default = 100,                                                                                            type = int,
                        help = 'Number of update steps between two logs if logging_strategy="steps"')
    
    parser.add_argument('--weight_decay',                  default = 0.1,                                                                                            type = float,
                        help = 'weight decay in the training')
    
    parser.add_argument('--warmup_steps',                  default = 1000,                                                                                           type = int,
                        help = 'steps of warm up of optimizer in training')
    
    parser.add_argument('--lr_scheduler_type',             default = 'cosine',                                                                                       type = str,
                        help = 'choose your learning rate schedular type:e.g., linear,consine etc.')
    

    parser.add_argument('--lr',                            default = 1e-4,                                                                                           type = float,
                        help = 'learning rate for model fine tune')
    
    parser.add_argument('--save_total_limit',              default = 10,                                                                                             type = int,
                        help = 'how many models you want to save')
    
    parser.add_argument('--save_steps',                    default = 50000,                                                                                          type = int,
                        help = 'how many models you want to save')
    
    parser.add_argument("--push2hub",                      default = False,                                                                                          type = str2bool, 
                        nargs = '?',  const = False,
                        help = "push the model to hub")
    
    parser.add_argument("--remove_unused_columns",         default = False,                                                                                          type = str2bool, 
                        nargs = '?',  const = False,
                        help = "Whether or not to automatically remove the columns unused by the model forward method")
    
    parser.add_argument("--ignore_data_skip",              default = True,                                                                                           type = str2bool, 
                        nargs = '?',  const = False,
                        help = ' When resuming training, whether or not to skip the epochs and batches to get the data loading at the same stage as in the previous training. If set to True, the training will begin faster (as that skipping step can take a long time) but will not yield the same results as the interrupted training would have.')
    

    parser.add_argument("--dataloader_pin_memory",         default = False,                                                                                          type = str2bool, 
                        nargs = '?',  const = False,
                        help = "Whether you want to pin memory in data loaders or not. Will default to False")
    
    args = parser.parse_args()
    return args




if __name__ == '__main__':

    
    
    
    # parse the arguments
    config = ini_config()
    print(config.lr)