import time
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, logging as transformers_logging
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
print(f"Before setting: EOS Token: {tokenizer.eos_token}, EOS Token ID: {tokenizer.eos_token_id}")
if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    model.resize_token_embeddings(len(tokenizer)) 
    print(f"After setting: EOS Token: {tokenizer.eos_token}, EOS Token ID: {tokenizer.eos_token_id}")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
print(f"Pad Token: {tokenizer.pad_token}, Pad Token ID: {model.config.pad_token_id}")
for param in model.parameters():
    param.requires_grad = False
num_prompts = 5
embedding_size = model.config.n_embd 
class SoftPromptEmbedding(nn.Module):
    def __init__(self, num_prompts, embedding_size):
        super(SoftPromptEmbedding, self).__init__()
        self.soft_prompt_embeddings = nn.Embedding(num_prompts, embedding_size)
    
    def forward(self, batch_size):
        soft_prompts = self.soft_prompt_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return soft_prompts

soft_prompt_layer = SoftPromptEmbedding(num_prompts, embedding_size).to(device)
optimizer = torch.optim.AdamW(soft_prompt_layer.parameters(), lr=5e-5)
train_inputs_subset = train_inputs.sample(frac=0.1, random_state=42)
train_outputs_subset = train_outputs.loc[train_inputs_subset.index]

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

def train_with_soft_prompts(model, soft_prompt_layer, train_loader, val_loader, optimizer, device, num_epochs=3, accumulation_steps=4):
    scaler = torch.amp.GradScaler()  
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            input_embeddings = model.transformer.wte(input_ids)
            batch_size = input_ids.size(0)
            soft_prompts = soft_prompt_layer(batch_size)
            inputs_with_prompts = torch.cat([soft_prompts, input_embeddings], dim=1)
            labels = torch.cat([torch.full((batch_size, soft_prompts.size(1)), -100, dtype=torch.long, device=device), labels], dim=1)
            with torch.amp.autocast(device_type='cuda'):  
                outputs = model(inputs_embeds=inputs_with_prompts, labels=labels)
                loss = outputs.loss / accumulation_steps 
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_train_loss += loss.item() * accumulation_steps 
        avg_train_loss = total_train_loss / len(train_loader)
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    input_embeddings = model.transformer.wte(input_ids)
                    batch_size = input_ids.size(0)
                    soft_prompts = soft_prompt_layer(batch_size)
                    inputs_with_prompts = torch.cat([soft_prompts, input_embeddings], dim=1)
                    labels = torch.cat([torch.full((batch_size, soft_prompts.size(1)), -100, dtype=torch.long, device=device), labels], dim=1)
                    with torch.amp.autocast(device_type='cuda'): 
                        outputs = model(inputs_embeds=inputs_with_prompts, labels=labels)
                        loss = outputs.loss
                    total_val_loss += loss.item()
        
            avg_val_loss = total_val_loss / len(val_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

def evaluate_model(model, soft_prompt_layer, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            input_embeddings = model.transformer.wte(input_ids)
            batch_size = input_ids.size(0)
            soft_prompts = soft_prompt_layer(batch_size)
            inputs_with_prompts = torch.cat([soft_prompts, input_embeddings], dim=1)
            labels = torch.cat([torch.full((batch_size, soft_prompts.size(1)), -100, dtype=torch.long, device=device), labels], dim=1)
            with torch.cuda.amp.autocast():
                outputs = model(inputs_embeds=inputs_with_prompts, labels=labels)
                loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

def calculate_rouge(model, soft_prompt_layer, tokenizer, dataloader, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            batch_size = input_ids.size(0)
            input_embeddings = model.transformer.wte(input_ids)
            soft_prompts = soft_prompt_layer(batch_size)
            inputs_with_prompts = torch.cat([soft_prompts, input_embeddings], dim=1)
            attention_mask = torch.ones(inputs_with_prompts.size()[:2], dtype=torch.long, device=device)
            summary_ids = model.generate(inputs_embeds=inputs_with_prompts,attention_mask=attention_mask,max_new_tokens=50,num_beams=1,early_stopping=True)
            generated_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
            prompt_length = soft_prompt_layer.soft_prompt_embeddings.num_embeddings
            labels = labels[:, prompt_length:]
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

def generate_summary_with_soft_prompts(input_text, model, soft_prompt_layer, tokenizer, device, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_embeddings = model.transformer.wte(input_ids)
    batch_size = input_ids.size(0)
    soft_prompts = soft_prompt_layer(batch_size)
    inputs_with_prompts = torch.cat([soft_prompts, input_embeddings], dim=1)
    attention_mask = torch.ones(inputs_with_prompts.size()[:2], dtype=torch.long, device=device)
    summary_ids = model.generate(inputs_embeds=inputs_with_prompts,attention_mask=attention_mask,max_new_tokens=max_length,num_beams=1, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_training_time(model, soft_prompt_layer, train_loader, val_loader, optimizer, device, num_epochs=3):
    start_time = time.time()
    train_with_soft_prompts(model, soft_prompt_layer, train_loader, val_loader, optimizer, device, num_epochs)
    end_time = time.time()
    return end_time - start_time

training_time = measure_training_time(model, soft_prompt_layer, train_loader, val_loader, optimizer, device, num_epochs=3)
print(f"Training Time: {training_time:.2f} seconds")
test_loss = evaluate_model(model, soft_prompt_layer, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
rouge1, rouge2, rougeL = calculate_rouge(model, soft_prompt_layer, tokenizer, test_loader, device)
print(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")
added_params = count_trainable_params(soft_prompt_layer)
print(f"Added Trainable Parameters: {added_params}")
sample_article = test_inputs.iloc[0]
print("Generated Summary:")
print(generate_summary_with_soft_prompts(sample_article, model, soft_prompt_layer, tokenizer, device))
def save_model_and_prompts_single_file(model, soft_prompt_layer, tokenizer, file_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'soft_prompt_state_dict': soft_prompt_layer.state_dict(),
        'tokenizer': tokenizer
    }, file_path)
    print(f"Model, tokenizer, and soft prompts saved in '{file_path}' as a .pth file")
file_path = './gpt2_with_soft_prompts.pth'
save_model_and_prompts_single_file(model, soft_prompt_layer, tokenizer, file_path)