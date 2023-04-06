'''
accelerate launch train.py \
--config_file /home/ypl/Research2022_CKQ/LLaMa-6B/ChatGLM-finetune-LoRA/ckq_config.yaml \
'''


### Load Model From huggingface

import os
import tqdm
import joblib
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

import peft
import loralib as lora
from peft import LoraConfig

import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup
# from tokenization_chatglm import ChatGLMTokenizer
import tokenization_chatglm
# import transformers_modules
# from tokenizaion

checkpoint = "THUDM/chatglm-6b"
mixed_precision = 'bf16'

accumulate_step = 8
MAX_LENGTH = 650

# from peft import LoraConfig,get_peft_model
config = LoraConfig(
    peft_type="LORA", 
    r=32, 
    lora_alpha=32, 
    target_modules=["q", "k", "v"],
    lora_dropout=0.1, 
)
'''
Args:
    r (int): Lora attention dimension
    target_modules (Union[List[str],str]): The names of the modules to apply Lora to.
    lora_alpha (float): The alpha parameter for Lora scaling.
    lora_dropout (float): The dropout probability for Lora layers.
    merge_weights (bool): Whether to merge the weights of the Lora layers with the base transformer model in eval mode.
    fan_in_fan_out (bool): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    enable_lora ( List[bool]): Used with lora.MergedLinear.
    bias (str): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
    modules_to_save (List[str]):List of modules apart from LoRA layers to be set as trainable
        and saved in the final checkpoint.
'''

LR = 2e-5
LR = 8e-5

NUM_EPOCHS = 2
warm_up_ratio = 0.1


# chatGLMTokenizer=ChatGLMTokenizer("THUDM/chatglm-6b/ice_text.model")
# chatGLMTokenizer=ChatGLMTokenizer("./vocab/ice_text.model")
# important! tokennizer
# 
tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)
# tokenizer.save_pretrained('./tokenizer')
model = AutoModel.from_pretrained(checkpoint,trust_remote_code=True)
# model.save_pretrained('./model')
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)# zero 2
# 精度控制
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step, deepspeed_plugin=deepspeed_plugin)

device = accelerator.device

### print setting

max_memory=accelerate.utils.get_max_memory()
accelerator.print("max_memory",max_memory)

### Insert LoRA to model
# LORa 分层
class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features//3)
        self.linear_k = torch.nn.Linear(in_features, out_features//3)
        self.linear_v = torch.nn.Linear(in_features, out_features//3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features//3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features//3].data
        self.linear_k.weight.data = target_layer.weight[target_layer.out_features//3:target_layer.out_features//3*2, :].data
        self.linear_k.bias.data = target_layer.bias[target_layer.out_features//3:target_layer.out_features//3*2].data
        self.linear_v.weight.data = target_layer.weight[target_layer.out_features//3*2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features//3*2:].data

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q,k,v], dim = -1)


for key, module in model.named_modules():
    if key.endswith('attention'):
        # merge layer！
        if isinstance(module.query_key_value, peft.tuners.lora.LoraModel):
            module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value.model)
        else:
            # 在这里切分
            # Here we split the query_key_value layer into three linear layer for LoRA. But you can also use merged linear.
            qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features) 
            qkv_layer.update(module.query_key_value)
            module.query_key_value = qkv_layer
            module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)


lora.mark_only_lora_as_trainable(model)# 只设置lora 是trainable的
model_parameters = filter(lambda p: p.requires_grad, model.parameters())# 模型参数

trainable_params = sum([np.prod(p.size()) for p in model_parameters])#
model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
non_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print('trainable_params:{} (percent{:.2f}%), non_trainable_params:{}'.format(trainable_params, trainable_params/non_trainable_params*100,non_trainable_params))


### Dataset

EOS_ID = 150005
### 这是对instruction 和input的处理，需要 {}
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

with open('data/alpaca_data.json', 'r') as f:
    content = json.load(f)


pairs = []
'''
使用format_map对字符串进行映射
'''
accelerator.print(f"dataset {len(content)}")
for line in content:
    if line['input'] == '':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(line)
    completion = line['output']+'</s>'
    if len(prompt) + len(completion) < MAX_LENGTH:
        pairs.append({'prompt':prompt, 'completion':completion})


class AlpacaDataset(Dataset):
    '''item& len'''
    def __init__(self, pairs, tokenizer) -> None:
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
 
    def __getitem__(self, index):
        if self.pairs[index]['completion'][-4:] == '</s>':
            prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
            completion = self.tokenizer.encode(self.pairs[index]['completion'][:-4], add_special_tokens=False)
            completion += [EOS_ID]
        else:
            prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
            completion = self.tokenizer.encode(self.pairs[index]['completion'], add_special_tokens=False)

        return {'prompt':prompt, 'completion':completion}

    def __len__(self):
        return len(self.pairs)


# fn
def collate_fn(batch):
    input_ids = []
    labels = []
    position_ids = []
    # 最大长度
    _max_length = max([len(obj['prompt'])+len(obj['completion']) for obj in batch])
    attention_mask = torch.ones((len(batch), _max_length, _max_length), device=device)
    # attention_mask
    attention_mask.tril_()

    for i, obj in enumerate(batch):
        context_length = obj['prompt'].index(150004)
        attention_mask[i, :, :context_length] = 1

        to_pad = _max_length - len(obj['prompt']) - len(obj['completion'])

        input_ids.append(obj['prompt'] + obj['completion'] + [tokenizer.pad_token_id] * to_pad)

        position_ids.append(torch.stack([torch.arange(0, _max_length, device=device), 
                                         torch.concat([torch.zeros(context_length - 1, device=device), 
                                                       torch.arange(0, _max_length - context_length + 1, device=device)])]).long())

        labels.append(torch.tensor([-100] * len(obj['prompt']) + 
                                   obj['completion'] +
                                   [-100] * to_pad, device=device).long())

    
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool() #bool设置
    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': attention_mask, 
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}

            
# load Alpaca dataset
train_dataset = AlpacaDataset(pairs,tokenizer=tokenizer,)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, shuffle=True, batch_size=1)


### Training 
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
## AdamW ADDS AN L2 regularization term to control the size of model parameters
### 重要 lr的更改
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(int(len(train_dataloader) / accumulate_step) * NUM_EPOCHS),
)
# 转化
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
model.to(device).train()


for epoch in range(NUM_EPOCHS):
    total_loss = 0
    
    for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
        with accelerator.accumulate(model):# 如果要梯度累加就得这样写
            outputs = model(**batch)
            loss_detach = outputs.loss.detach().cpu().float()
            # 每个轮次
            t.set_description(f"loss: {loss_detach}")
            total_loss += loss_detach
            loss = outputs.loss
            accelerator.backward(loss) # 梯度平均
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if step!=0 and step%10000==0 and accelerator.is_main_process:
            peft_model_id = f"finetune_{epoch}_{step}"
            accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), '/home/ypl/Research2022_CKQ/LLaMa-6B/ChatGLM-finetune-LoRA/saved/ckq_'+peft_model_id+'.pt')
            
        

    accelerator.wait_for_everyone() ## 等待其他机子
    if accelerator.is_main_process:
        peft_model_id = f"finetune_{epoch}"
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), '/home/ypl/Research2022_CKQ/LLaMa-6B/ChatGLM-finetune-LoRA/saved/ckq_'+peft_model_id+'.pt')
    
