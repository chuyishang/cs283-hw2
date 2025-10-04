import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tokenizers import ByteLevelBPETokenizer
from tokenizers.normalizers import Lowercase

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load the corpus
with open('data/stories.txt', 'r', encoding='utf-8') as f:
    en_corpus = [line.strip() for line in f.readlines()]


def get_tokenizer(data_path: str, vocab_size=10000):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.normalizer = Lowercase()

    tokenizer.train(files=data_path, vocab_size=vocab_size, min_frequency=2)

    return tokenizer

en_tokenizer = get_tokenizer("data/stories.txt")
en_tokenized = [en_tokenizer.encode(s).ids for s in en_corpus]

def get_bow_context(tokenized_sentences: List[List[str]]) -> List[Tuple[int, int]]:
    """
    Creates positive (target, context) examples where context is any other word in the sentence.
    NOTE: this is modified to return ints, as according to the Ed clarifications.
    """
    # TODO: Implement this function
    out_context = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence)):
            for j in range(len(sentence)):
                if i != j:
                    out_context.append((sentence[i], sentence[j]))
    return out_context


def get_neighbor_context(tokenized_sentences: List[List[int]], window_size: int) -> List[Tuple[int, int]]:
    """
    Creates positive (target, context) examples from tokens within a given window size.
    """
    # TODO: Implement this function
    out_context = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence)):
            for j in range(1, window_size+1):
                ctx_left_idx = i - j
                ctx_right_idx = i + j
                if ctx_left_idx >= 0:
                    out_context.append((sentence[i], sentence[ctx_left_idx]))
                if ctx_right_idx < len(sentence):
                    out_context.append((sentence[i], sentence[ctx_right_idx]))
    return out_context

def get_context_distribution(pairs: List[Tuple[int, int]], vocab_size: int) -> List[float]:
    """
    Computes the probability distribution over contexts P(c).

    Args:
        pairs: A list of (token, context) pairs.
        vocab_size: The size of the vocabulary.

    Returns:
        A list of probabilities for each context token, ordered by token index.
    """
    # TODO: Implement this function
    context_distribution = [0] * vocab_size
    for pair in pairs:
        context_distribution[pair[1]] += 1
    # context_distribution = [x / len(pairs) for x in context_distribution]
    return torch.tensor(context_distribution)


def sample_negative_contexts(context_distribution: torch.Tensor, num_samples: int, batch_size=1) -> torch.Tensor:
    """
    Samples negative contexts based on the context distribution.
    """
    # TODO: Implement this function
    context_distribution = context_distribution / context_distribution.sum()
    # batched_distribution = context_distribution.repeat(batch_size, 1)
    batched_distribution = context_distribution.expand(batch_size, -1)
    return torch.multinomial(batched_distribution, num_samples, replacement=True)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGramModel, self).__init__()
        # TODO: Implement the model layers
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        # TODO: Implement the forward pass
        # targets should be (B, 1)
        # context should be (B, K)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        target_embeds = self.target_embeddings(target) # (B, 1, D)
        context_embeds = self.output_embeddings(context) # (B, K, D)
        # return torch.sum(target_embeds.unsqueeze(1) * context_embeds, dim=-1) # (B, K)
        return torch.sum(target_embeds * context_embeds, dim=-1) # (B, K)
        
from tqdm import tqdm
device = 'cuda'

class SkipGramDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def train_skipgram(corpus, tokenizer, num_epochs=30):

    # Initialize the model, loss function, optimizer, and dataloader

    embedding_dim = 128
    vocab_size = len(tokenizer.get_vocab())
    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    batch_size = 2**16

    neighbor_pairs = get_neighbor_context([tokenizer.encode(s).ids for s in corpus], window_size=5)
    dataset = SkipGramDataset(neighbor_pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )    

    # Training loop
    # TODO: Implement this. 
    # You will need to combine positive and negative samples in the loss computation, and adjust the labels accordingly.
    NUM_NEGATIVE_SAMPLES = 31
    context_dist = get_context_distribution(neighbor_pairs, vocab_size)

    # if not os.path.exists('cached_neg_ctx.pt'):
        # print("CACHING")
        # cached_neg_ctx = []
        # for step, batch in enumerate(tqdm(dataloader)):
            # neg_ctx = sample_negative_contexts(context_dist, NUM_NEGATIVE_SAMPLES, batch_size) # (B, K-1)
            # cached_neg_ctx.append(neg_ctx)
        # cached_neg_ctx = torch.stack(cached_neg_ctx, dim=0)
        # torch.save(cached_neg_ctx, 'cached_neg_ctx.pt')
        # print("finished caching") 
        # print(f"cached_neg_ctx.shape: {cached_neg_ctx.shape}")
    # else:
        # cached_neg_ctx = torch.load('cached_neg_ctx.pt')
        # print("loaded cached negative contexts")
        # print(f"cached_neg_ctx.shape: {cached_neg_ctx.shape}")



    model.train()
    step = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            target, positive_context = batch

            # cached_neg_ctx_step = cached_neg_ctx[torch.randperm(cached_neg_ctx.size(0))]
            # negative_contexts = cached_neg_ctx[step]
            negative_contexts = sample_negative_contexts(context_dist, NUM_NEGATIVE_SAMPLES, target.shape[0]) # (B, K-1)

            target = target.unsqueeze(1).to(device) # (B, 1)
            positive_context = positive_context.unsqueeze(1) # (B, 1)
            full_context = torch.cat([positive_context, negative_contexts], dim=-1).to(device) # (B, K)

            scores = model(target, full_context) # (B, K)
            labels = torch.zeros_like(scores, dtype=torch.float, device=device)
            labels[:, 0] = 1.0
            labels = labels.to(device)
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Loss: {loss.item()}")
        print(f"EPOCH {epoch+1} COMPLETED")
    return model


device = 'cuda'


model = train_skipgram(en_corpus, en_tokenizer, num_epochs=30)
# model = train_skipgram(en_corpus, en_tokenizer, num_epochs=1)
torch.save(model.state_dict(), 'results/skipgram_model.pth')
# with open('results/skipgram_model.pth', 'rb') as f:
    # model.load_state_dict(torch.load(f))
    # model.to(device)
    # model.eval()

test_center_word = torch.tensor(np.load('data/center_word.npy')).to(device)
test_context_words = torch.tensor(np.load('data/context_words.npy')).to(device)
print(test_center_word.shape, test_context_words.shape)

test_scores = model(test_center_word, test_context_words)
np.save('results/skipgram_scores.npy', test_scores.cpu().detach().numpy())