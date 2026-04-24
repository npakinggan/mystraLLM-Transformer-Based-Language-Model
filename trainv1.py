<<<<<<< HEAD
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import re

# hyperparameters
batch_size = 64
block_size = 256 # was 256
max_iters = 5000 # was 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embd = 384 # was 384
n_head = 6 
n_layer = 6 # was 6
dropout = .3 # originally .2, raised to .3 for shorter data, # preventing overfitting 

torch.manual_seed(1337)


# loading and wrangling json of my messages 
df = pd.read_json('messages.json')
df = df[df['Contents'].notna() & (df['Contents'].str.strip() != '')]
df = df.sort_values('Timestamp')

def is_only_url(s):
    return bool(re.fullmatch(r'https?://\S+', s.strip()))

def clean_message(s):
    s = re.sub(r'https?://\S+', '', s)  # remove URLs
    s = s.strip()
    return s

df = df[~df['Contents'].apply(is_only_url)]
text = '\n'.join(df['Contents'].astype(str).tolist())


df['Contents'] = df['Contents'].apply(clean_message)
df = df[df['Contents'] != '']


# all unique characters in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers for tokenization
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# input: string, output a list of integers
encode = lambda s: [stoi[c] for c in s]

# input: list of integers, output a string 
decode = lambda l: ''.join([itos[i] for i in l])

# training and testing splits
data = torch.tensor(encode(text), dtype = torch.long)

# splitting data into training and validation sets
n = int(0.9*len(data)) # first 90% of the data will be used to train, rest val
train_data = data[:n]
val_data = data[n:]

# loading data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B,T,C = x.shape
        # k and q both (B,T,C)
        k = self.key(x) 
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B,T,T)
        wei = self.dropout(wei)

        # weighted aggregation of values
        v = self.value(x) # (B,T,C)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of attention running in parallel!"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out) # added residual connections
        return out

class FeedForward(nn.Module):
    """linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # added residual connections
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # added residual connections
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """transformer block: commm. followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        
        # layer norms added
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # added residual connections and layer norms
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))        
        return x


# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off logits for next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.blocks = nn.Sequential(
        #    Block(n_embd, n_head = 4),
        #    Block(n_embd, n_head = 4),
        #    Block(n_embd, n_head = 4),
        #    nn.LayerNorm(n_embd),
        #)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape

        # idx and targets are (B,T) tensor of ints
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # changing scope to last block_size elements
            idx_cond = idx[:, -block_size:]
            # predict
            logits, loss = self(idx_cond)
            # focus on last step
            logits = logits[:, -1, :] # (B,C)
            # getting probabilities
            probs = F.softmax(logits, dim = 1) # (B,C) 
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # eval the loss on train and val sets
    if (iter % eval_interval) == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    # sample batch
    xb, yb = get_batch('train')

    # loss eval
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from model
context = torch.zeros((1,1), dtype = torch.long, device = device)
generated = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated)

# saving result
with open('output.txt', 'w') as file:
    file.write(generated)

torch.save(m.state_dict(), 'neil_model.pth')
=======
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import re

# hyperparameters
batch_size = 64
block_size = 256 # was 256
max_iters = 5000 # was 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embd = 384 # was 384
n_head = 6 
n_layer = 6 # was 6
dropout = .3 # originally .2, raised to .3 for shorter data, # preventing overfitting 

torch.manual_seed(1337)


# loading and wrangling json of my messages 
df = pd.read_json('messages.json')
df = df[df['Contents'].notna() & (df['Contents'].str.strip() != '')]
df = df.sort_values('Timestamp')

def is_only_url(s):
    return bool(re.fullmatch(r'https?://\S+', s.strip()))

def clean_message(s):
    s = re.sub(r'https?://\S+', '', s)  # remove URLs
    s = s.strip()
    return s

df = df[~df['Contents'].apply(is_only_url)]
text = '\n'.join(df['Contents'].astype(str).tolist())


df['Contents'] = df['Contents'].apply(clean_message)
df = df[df['Contents'] != '']


# all unique characters in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers for tokenization
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# input: string, output a list of integers
encode = lambda s: [stoi[c] for c in s]

# input: list of integers, output a string 
decode = lambda l: ''.join([itos[i] for i in l])

# training and testing splits
data = torch.tensor(encode(text), dtype = torch.long)

# splitting data into training and validation sets
n = int(0.9*len(data)) # first 90% of the data will be used to train, rest val
train_data = data[:n]
val_data = data[n:]

# loading data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B,T,C = x.shape
        # k and q both (B,T,C)
        k = self.key(x) 
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B,T,T)
        wei = self.dropout(wei)

        # weighted aggregation of values
        v = self.value(x) # (B,T,C)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of attention running in parallel!"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out) # added residual connections
        return out

class FeedForward(nn.Module):
    """linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # added residual connections
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # added residual connections
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """transformer block: commm. followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        
        # layer norms added
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # added residual connections and layer norms
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))        
        return x


# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off logits for next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.blocks = nn.Sequential(
        #    Block(n_embd, n_head = 4),
        #    Block(n_embd, n_head = 4),
        #    Block(n_embd, n_head = 4),
        #    nn.LayerNorm(n_embd),
        #)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape

        # idx and targets are (B,T) tensor of ints
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # changing scope to last block_size elements
            idx_cond = idx[:, -block_size:]
            # predict
            logits, loss = self(idx_cond)
            # focus on last step
            logits = logits[:, -1, :] # (B,C)
            # getting probabilities
            probs = F.softmax(logits, dim = 1) # (B,C) 
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # eval the loss on train and val sets
    if (iter % eval_interval) == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    # sample batch
    xb, yb = get_batch('train')

    # loss eval
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from model
context = torch.zeros((1,1), dtype = torch.long, device = device)
generated = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated)

# saving result
with open('output.txt', 'w') as file:
    file.write(generated)

torch.save(m.state_dict(), 'neil_model.pth')
>>>>>>> 9e7be52 (first commit)
