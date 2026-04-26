from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')
model.eval()

model = model.to(device)

def generate(prompt="", length=60, variations=3):
    full_prompt = f"{prompt} <|sep|>"

    token_ids = tokenizer.encode(full_prompt)
    input_ids = torch.tensor([token_ids]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    outputs = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens = length,
        num_return_sequences = variations,
        do_sample =True,
        temperature = .65,
        top_p = .95,
        top_k = 40,
        repetition_penalty = 1.3,
        no_repeat_ngram_size = 3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )

    for i, output in enumerate(outputs):
        # decode the tokens
        text = tokenizer.decode(output, skip_special_tokens=True)
        
        # take prompt away
        if '<|sep|>' in text:
            text = text.split('<|sep|>')[-1].strip()

        # removing @ and mentions
        text = re.sub(r'<@\d+>', '', text).strip()
        
        # cutting newline and printing response
        text = text.split('\n')[0].strip()
        print(f'\n--- {i+1} ---')
        print(text)

# little prompt to ask
generate(prompt="")