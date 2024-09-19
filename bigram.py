import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iterations = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iterations = 200
num_embedding = 64
num_head = 4
num_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('Nano GPT\input.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# here are all the unique characters that occur in this text
characters = sorted(list(set(text_data)))
vocabulary_size = len(characters)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(characters) }
itos = { i:ch for i,ch in enumerate(characters) }
encoder = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decoder = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encoder(text_data), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def CreateBatch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = CreateBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output

class Head(nn.Module):
    """ Implementation of single head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embedding, head_size, bias=False)
        self.query = nn.Linear(num_embedding, head_size, bias=False)
        self.value = nn.Linear(num_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #Buffers are saved and loaded with the model's state but are not updated during backpropagation.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T), scaling factor (1/sqrt(C))is used to normalize the output of the dot-product attention
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        output = weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return output

class MultiHeadAttention(nn.Module):
    """ Implementation of multiple heads of self-attention executed parallelly """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projections = nn.Linear(num_embedding, num_embedding) #linear transformations applied to the input data to create q,k,v
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projections(out))
        return out

class FeedFoward(nn.Module):
    """ Implementation of a simple sequential net of layers """

    def __init__(self, num_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embedding, 4 * num_embedding),
            nn.ReLU(),
            nn.Linear(4 * num_embedding, num_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_embedding, num_head):
        # num_embedding: embedding dimension, num_head: the number of heads we'd like
        super().__init__()
        head_size = num_embedding // num_head
        self.sa = MultiHeadAttention(num_head, head_size)
        self.ffwd = FeedFoward(num_embedding)
        self.ln1 = nn.LayerNorm(num_embedding)
        self.ln2 = nn.LayerNorm(num_embedding)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, num_embedding) # maps each tokento a vector of size 'num_embedding'
        self.position_embedding_table = nn.Embedding(block_size, num_embedding) # positional information for each token in the sequence
        self.blocks = nn.Sequential(*[Block(num_embedding, num_head=num_head) for _ in range(num_layer)]) # stack of Transformer blocks 
        self.ln_f = nn.LayerNorm(num_embedding) # final layer norm to normalize the activations and help stabilize training
        self.lm_head = nn.Linear(num_embedding, vocabulary_size) # projects the embeddings from the final block to a vector of size vocabulary_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) shape is broadcasted across all batches
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocabulary_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generates logits for the next token prediction """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # # focus only on the last time step, becomes (B, C)
            probabilities = F.softmax(logits, dim=-1) # # apply softmax to get probabilities -> (B, C)
            idx_next = torch.multinomial(probabilities, num_samples=1) # # sample from the distribution -> (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # # append sampled index to the running sequence -> (B, T+1)
        return idx

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, num_embedding)
        self.position_embedding_table = nn.Embedding(block_size, num_embedding)
        self.blocks = nn.Sequential(*[Block(num_embedding, num_head=num_head) for _ in range(num_layer)])
        self.ln_f = nn.LayerNorm(num_embedding) # final layer norm
        self.lm_head = nn.Linear(num_embedding, vocabulary_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ This initialization helps in breaking the symmetry and ensuring that the weights start with random values that are close to zero but not exactly zero. 
            This prevents issues like dead neurons (in case of ReLU activations) and helps in faster convergence."""
            
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) #initializes the weights of the linear layer using a normal (Gaussian) distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # initializes the bias of the linear layer to zero
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocabulary_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Generates logits for the next token prediction """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # # focus only on the last time step, becomes (B, C)
            probabilities = F.softmax(logits, dim=-1) # # apply softmax to get probabilities -> (B, C)
            idx_next = torch.multinomial(probabilities, num_samples=1) # # sample from the distribution -> (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # # append sampled index to the running sequence -> (B, T+1)
        return idx

ch = int(input('Which model do you want to use (1 or 2):\n1. BigramLanguageModel\n2. GPTLanguageModel\n'))
model = BigramLanguageModel() if ch == 1 else GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6,'Million parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW is a variation of the Adam optimizer with weight decay (L2 regularization) to prevent overfitting

for iter in range(max_iterations):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iterations - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = CreateBatch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # clears the gradients of the model's parameters. Without this step, gradients would accumulate across iterations, leading to incorrect updates.
    loss.backward()
    optimizer.step() # applies the computed gradients to update the model's parameters (weights) according to the AdamW optimization algorithm. The optimizer adjusts the parameters to minimize the loss.

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # (B, T) -> (1, 1)
print(decoder(m.generate(context, max_new_tokens=2000)[0].tolist())) 