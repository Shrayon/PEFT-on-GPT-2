import time
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

df1 = pd.read_csv('/kaggle/input/text-summarisation-dataset/cnn_dailymail/train.csv')
df2 = pd.read_csv('/kaggle/input/text-summarisation-dataset/cnn_dailymail/validation.csv')
df3 = pd.read_csv('/kaggle/input/text-summarisation-dataset/cnn_dailymail/test.csv')
train_inputs = df1['article']
train_outputs = df1['highlights']
val_inputs = df2['article']
val_outputs = df2['highlights']
test_inputs = df3['article']
test_outputs = df3['highlights']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
for param in model.parameters():
    param.requires_grad = False
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(r=16,lora_alpha=32,target_modules=["attn.c_proj", "mlp.c_fc"], lora_dropout=0.1,bias="none",task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

added_params = count_trainable_params(model)
print(f"Added Trainable Parameters: {added_params}")
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
train_inputs_subset = train_inputs.sample(frac=0.1, random_state=42).reset_index(drop=True)
train_outputs_subset = train_outputs.loc[train_inputs_subset.index].reset_index(drop=True)

class TextSummaryDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=256):
        self.inputs = inputs.reset_index(drop=True)
        self.outputs = outputs.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs.iloc[idx]
        output_text = self.outputs.iloc[idx]
        input_encoding = self.tokenizer(input_text,max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt')
        output_encoding = self.tokenizer(output_text,max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt')
        input_ids = input_encoding['input_ids'].squeeze(0)
        output_ids = output_encoding['input_ids'].squeeze(0)
        return input_ids, output_ids
    
train_dataset = TextSummaryDataset(train_inputs_subset, train_outputs_subset, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataset = TextSummaryDataset(val_inputs, val_outputs, tokenizer)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = torch.utils.data.DataLoader(TextSummaryDataset(test_inputs, test_outputs, tokenizer), batch_size=2, shuffle=False)

def train_with_lora(model, train_loader, val_loader, optimizer, device, num_epochs=3, accumulation_steps=4):
    scaler = torch.cuda.amp.GradScaler()  
    start_time = time.time()
    max_gpu_memory = 0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps 
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_train_loss += loss.item() * accumulation_steps 
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.max_memory_allocated(device)
                max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
        avg_train_loss = total_train_loss / len(train_loader)
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"Peak GPU Memory Usage: {max_gpu_memory / 1e6:.2f} MB")
    return training_time, max_gpu_memory

training_time, max_gpu_memory = train_with_lora(model, train_loader, val_loader, optimizer, device, num_epochs=3)
added_params = count_trainable_params(model)
print(f"Added Trainable Parameters: {added_params}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Peak GPU Memory Usage: {max_gpu_memory / 1e6:.2f} MB")

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

test_loss = evaluate_model(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")

def calculate_rouge(model, tokenizer, dataloader, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            generated_ids = model.generate(input_ids=input_ids,max_new_tokens=50,num_beams=4,early_stopping=True,no_repeat_ngram_size=2)
            generated_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            reference_summaries = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
            for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
                scores = scorer.score(ref_summary, gen_summary)
                rouge1_list.append(scores['rouge1'].fmeasure)
                rouge2_list.append(scores['rouge2'].fmeasure)
                rougeL_list.append(scores['rougeL'].fmeasure)
    avg_rouge1 = np.mean(rouge1_list)
    avg_rouge2 = np.mean(rouge2_list)
    avg_rougeL = np.mean(rougeL_list)
    return avg_rouge1, avg_rouge2, avg_rougeL

def generate_summary_with_attention(input_text, model, tokenizer, device, max_length=50):
    model.eval()
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    summary_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=max_length,num_beams=4,early_stopping=True,no_repeat_ngram_size=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

rouge1, rouge2, rougeL = calculate_rouge(model, tokenizer, test_loader, device)
print(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")
sample_article = test_inputs.iloc[0]
print("\nSample Article:")
print(sample_article)
print("\nGenerated Summary:")
print(generate_summary_with_attention(sample_article, model, tokenizer, device))
def save_lora_model_pth(model, tokenizer, file_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_state': tokenizer
    }, file_path)
    print(f"Model and tokenizer saved in '{file_path}' as a .pth file")
file_path = './gpt2_lora_model.pth'
save_lora_model_pth(model, tokenizer, file_path)