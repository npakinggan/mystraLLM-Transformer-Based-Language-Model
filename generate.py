from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')
model.eval()

model = model.to(device)

def generate(prompt='', length=80, variations=30):
    inputs = tokenizer.encode(prompt, return_tensors = 'pt').to(device)
    outputs = model.generate(
        inputs,
        max_length = length,
        num_return_sequences = variations,
        do_sample =True,
        temperature = .9,
        top_p = .95,
        top_k = 50,
        pad_token_id=tokenizer.eos_token_id
    )

    for i, output in enumerate(outputs):
        print(f'\n--- {i+1} ---')
        print(tokenizer.decode(output, skip_special_tokens=True))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')
model.eval()

model = model.to(device)

def generate(prompt='', length=80, variations=30):
    inputs = tokenizer.encode(prompt, return_tensors = 'pt').to(device)
    outputs = model.generate(
        inputs,
        max_length = length,
        num_return_sequences = variations,
        do_sample =True,
        temperature = .9,
        top_p = .95,
        top_k = 50,
        pad_token_id=tokenizer.eos_token_id
    )

    for i, output in enumerate(outputs):
        print(f'\n--- {i+1} ---')
        print(tokenizer.decode(output, skip_special_tokens=True))

generate(prompt='what should be your name?')