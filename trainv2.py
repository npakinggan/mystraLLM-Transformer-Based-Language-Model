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
import emoji

with open('combined_messages.json', 'r', encoding = 'utf-8') as f:
    data = json.load(f)

# loading tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def quality_message(s):
    # removing short or overly long messages
    if not s or len(s) < 8 or len(s) > 200:
        return False
    # removing @ or discord mentions in messages
    if re.fullmatch(r'\s*', s.strip()) or re.fullmatch(r'(<@\d+>\s*)+', s.strip()) or re.search(r'<@\d+>', s):  # skip if empty after URL removal
        return False
    # removing repetitive emojis (at least 3)    
    if sum(1 for c in s if c in emoji.EMOJI_DATA) > 3:
        return False
    # removing if repeating 50% more than words
    words = s.split()
    if len(words) > 3 and len(set(words)) / len(words) < 0.5:
        return False
    
    return True

entries = [e for e in data 
        if e.get('Contents', '').strip() 
        and not e.get("Attachments", '').strip() 
        and quality_message(e['Contents'].strip())]

messages = []

for i in range(len(entries) - 1):

    # stripping urls from responses
    prompt = re.sub(r'https?://\S+', '', entries[i]['Contents']).strip()
    response = re.sub(r'https?://\S+', '', entries[i+1]['Contents']).strip()

    if not prompt or not response:
        continue

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
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    fp16 = True,
    warmup_steps=200,
    weight_decay=.01
)

trainer = Trainer(model = model, 
    args = training_args,
    data_collator=collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')