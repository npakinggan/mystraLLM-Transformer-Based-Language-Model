import json
import re
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

with open('messages.json', 'r', encoding = 'utf-8') as f:
    data = json.load(f)

# loading tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

messages = []
entries = [e for e in data if e.get('Contents', '').strip()]

for i in range(len(entries) - 1):
    prompt = entries[i]['Contents'].strip()
    response = entries[i+1]['Contents'].strip()

    prompt = re.sub(r'https?://\S+', '<URL>', prompt)
    response = re.sub(r'https?://\S+', '<URL>', response)

    messages.append(f"{prompt} <|sep|> {response} {tokenizer.eos_token}")

# tokenizing dataset
def tokenize(batch):
    return tokenizer(
        batch['text'],
        truncation = True,
        max_length = 128,
        padding = 'max_length'
    )

dataset = Dataset.from_dict({'text': messages})
dataset = dataset.map(tokenize, batched = True, remove_columns=['text'])

# model
model = GPT2LMHeadModel.from_pretrained('gpt2')
collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)

# training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=250,
    save_total_limit=2,
    logging_steps=50,
    fp16 = True
)

trainer = Trainer(model = model, 
    args = training_args,
    data_collator=collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')