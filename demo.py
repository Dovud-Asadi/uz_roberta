import re
import os
import json
import string
import torch
import wandb
from datasets import Dataset
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForMaskedLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from huggingface_hub import login

login(token = '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')

# Initialize Weights & Biases for tracking
wandb.init(project="xlm-roberta-uzbek")


wandb.login(key='1db3b02d78973739be12be14efcbb1004e74bd2f')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = text.replace('‘', "'").replace('’', "'").replace('`', "'")
    text = re.sub(r'[^a-z\.\'\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and clean text
file_path = 'data/text.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

cleaned_text = clean_text(text)
sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
sentences = [sentence.strip() for sentence in sentences if len(sentence) > 0]

# Remove sentences with fewer than 20 characters
def remove_less_than_twenty(lst):
    return [x for x in lst if len(x) > 20]

sentences = remove_less_than_twenty(sentences)


# Create dataset
dataset = Dataset.from_dict({"text": sentences})

model_name = 'xlm-roberta-large'
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=False, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

model = XLMRobertaForMaskedLM.from_pretrained(model_name)

# Check if GPU is available
use_fp16 = torch.cuda.is_available()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results2",  
    overwrite_output_dir=True, 
    num_train_epochs=3,  
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=4,  
    save_steps=10_000, 
    save_total_limit=2,  
    fp16=use_fp16,  
    dataloader_num_workers=4,  
    logging_dir='./logs',  
    report_to="wandb",  
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

trainer.train()

model.save_pretrained("./uzbek_xlm_roberta_model2")
tokenizer.save_pretrained("./uzbek_xlm_roberta_tokenizer2")
