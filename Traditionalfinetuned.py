import time
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, logging as transformers_logging
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from rouge_score import rouge_scorer
import numpy as np
import gc
transformers_logging.set_verbosity_error()

torch.manual_seed(42)
np.random.seed(42)
df_train = pd.read_csv('/kaggle/input/text-summarisation-dataset/cnn_dailymail/train.csv')
df_val = pd.read_csv('/kaggle/input/text-summarisation-dataset/cnn_dailymail/validation.csv')
df_test = pd.read_csv('/kaggle/input/text-summarisation-dataset/cnn_dailymail/test.csv')
train_inputs = df_train['article']
train_outputs = df_train['highlights']
val_inputs = df_val['article']
val_outputs = df_val['highlights']
test_inputs = df_test['article']
test_outputs = df_test['highlights']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
for name, param in model.named_parameters():
    if not (name.startswith('transformer.h.10') or
            name.startswith('transformer.h.11') or
            name.startswith('lm_head')):
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f'Trainable Parameters: {trainable_params} / {total_params} ({(trainable_params/total_params)*100:.2f}%)')
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
train_subset_frac = 0.1
train_inputs_subset = train_inputs.sample(frac=train_subset_frac, random_state=42).reset_index(drop=True)
train_outputs_subset = train_outputs.loc[train_inputs_subset.index].reset_index(drop=True)

class TextSummaryDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=256):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        input_enc = self.tokenizer(input_text,max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt')
        output_enc = self.tokenizer(output_text,max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt')
        input_ids = input_enc['input_ids'].squeeze(0)
        attention_mask = input_enc['attention_mask'].squeeze(0)
        labels = output_enc['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

batch_size = 2 
train_dataset = TextSummaryDataset(train_inputs_subset, train_outputs_subset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TextSummaryDataset(val_inputs, val_outputs, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TextSummaryDataset(test_inputs, test_outputs, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
num_epochs = 10
gradient_accumulation_steps = 4
max_grad_norm = 1.0
early_stopping_patience = 3

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, accumulation_steps=4, max_grad_norm=1.0, early_stopping_patience=3):
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    max_gpu_memory = 0 
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_train_loss += loss.item() * accumulation_steps
            current_gpu_memory = torch.cuda.memory_allocated(device)
            max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
        torch.cuda.empty_cache()
        gc.collect()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")
    print(f"Peak GPU memory usage during training: {max_gpu_memory / 1e6:.2f} MB")
    return training_time

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    max_gpu_memory = 0  
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            total_loss += loss.item()
            current_gpu_memory = torch.cuda.memory_allocated(device)
            max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
    print(f"Peak GPU memory usage during evaluation: {max_gpu_memory / 1e6:.2f} MB")
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def calculate_rouge(model, tokenizer, dataloader, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    max_gpu_memory = 0  
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=50,num_beams=4,early_stopping=True,no_repeat_ngram_size=2)
            generated_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            reference_summaries = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
            current_gpu_memory = torch.cuda.memory_allocated(device)
            max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
            for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
                scores = scorer.score(ref_summary, gen_summary)
                rouge1_list.append(scores['rouge1'].fmeasure)
                rouge2_list.append(scores['rouge2'].fmeasure)
                rougeL_list.append(scores['rougeL'].fmeasure)
    print(f"Peak GPU memory usage during ROUGE calculation: {max_gpu_memory / 1e6:.2f} MB")
    avg_rouge1 = np.mean(rouge1_list)
    avg_rouge2 = np.mean(rouge2_list)
    avg_rougeL = np.mean(rougeL_list)
    return avg_rouge1, avg_rouge2, avg_rougeL


def generate_summary_with_attention(input_text, model, tokenizer, device, max_length=50):
    model.eval()
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    summary_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_length,num_beams=1,early_stopping=True,no_repeat_ngram_size=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

training_time = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,accumulation_steps=gradient_accumulation_steps,max_grad_norm=max_grad_norm, early_stopping_patience=early_stopping_patience)
print(f"Training Time: {training_time:.2f} seconds")
test_loss = evaluate_model(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
rouge1, rouge2, rougeL = calculate_rouge(model, tokenizer, test_loader, device)
print(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")
sample_article = test_inputs.iloc[0]
print("\nSample Article:")
print(sample_article)
print("\nGenerated Summary:")
print(generate_summary_with_attention(sample_article, model, tokenizer, device))
def save_model_pth(model, tokenizer, file_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, file_path)
    print(f"Model and tokenizer saved to {file_path}")
save_path = './gpt2_traditional_finetuned.pth'
save_model_pth(model, tokenizer, save_path)