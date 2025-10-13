import itertools
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from invasion_biology_embeddings.auxiliary import relative_path
import json
from transformers import AutoModel, AutoTokenizer
from invasion_biology_embeddings.generate_aspect_specific_summaries import prompts
from settings import encoder_checkpoint, proj_dimension

BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
LR = 1e-5

with open(relative_path("../data/mistral_invasion_biology_summaries.json"), "r") as f:
    dataset = json.load(f)

prompt_identifiers = list(set([p.split("_")[0] for p in prompts.keys()]))

print("Analyzing training data:")
print("\nNumber of valid samples:")
for p in prompt_identifiers:
    print(f"{p}: {len([x for x in dataset['unlabeled'].values() if len([y for y in itertools.chain.from_iterable([x[c] for c in x.keys() if p in c])
                                                                                if 'not applicable' not in y.lower()]) >= 2])}")
print("\nNumber of total summaries for valid samples:")
for p in prompt_identifiers:
    print(f"{p}: {sum([len([y for y in itertools.chain.from_iterable([x[c] for c in x.keys() if p in c])
                                                                                if 'not applicable' not in y.lower()]) for x in dataset['unlabeled'].values() if len([y for y in itertools.chain.from_iterable([x[c] for c in x.keys() if p in c])
                                                                                if 'not applicable' not in y.lower()]) >= 2])}")


def batched_similarity(embeddings1, embeddings2, batch_size=100):
    n1 = embeddings1.size(0)
    n2 = embeddings2.size(0)
    similarities = torch.zeros((n1, n2), device=embeddings1.device)

    for i in range(0, n1, batch_size):
        for j in range(0, n2, batch_size):
            batch1 = embeddings1[i:i + batch_size]
            batch2 = embeddings2[j:j + batch_size]

            sim = F.cosine_similarity(batch1.unsqueeze(1), batch2.unsqueeze(0), dim=-1)
            similarities[i:i + batch_size, j:j + batch_size] = sim

    return similarities


def evaluate_embedding_model(model, tokenizer, eval_data):
    model.eval()

    all_summaries = [sentence for pair in eval_data for sentence in pair]

    with torch.no_grad():
        summary_embeddings = []
        batch_size = BATCH_SIZE
        for i in range(0, len(all_summaries), batch_size):
            batch = all_summaries[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=510, return_tensors="pt").to("cuda")
            batch_embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
            summary_embeddings.append(batch_embeddings)

        all_summary_embeddings = torch.cat(summary_embeddings, dim=0)

        summary_similarities = batched_similarity(all_summary_embeddings, all_summary_embeddings)
        summary_similarities.fill_diagonal_(-torch.inf)
        arg_sorted = torch.argsort(summary_similarities, dim=-1, descending=True)
        rank_scores = []
        for i in range(all_summary_embeddings.shape[0]//2):
            correct_rank = (arg_sorted[i*2] == i*2+1).nonzero(as_tuple=True)[0].item()
            rank_scores.append(correct_rank)

            correct_rank = (arg_sorted[i*2+1] == i*2).nonzero(as_tuple=True)[0].item()
            rank_scores.append(correct_rank)
        average_summary_rank = np.mean(rank_scores)
        print(f"Average summary match rank: {average_summary_rank:.3f}")

        return -average_summary_rank

def train_embedding_model(prompt_code):
    random.seed(0)

    model = AutoModel.from_pretrained(encoder_checkpoint).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    available_data = [[x for x in itertools.chain.from_iterable([x[c] for c in x.keys() if prompt_code in c]) if "not applicable" not in x.lower()]
                      for x in dataset["unlabeled"].values()]
    available_data = [x for x in available_data if len(x) >= 2]

    random.shuffle(available_data)
    eval_data = [random.sample(x, 2) for x in available_data[:150]]
    train_data = available_data[150:]
    sampling_list = train_data.copy()

    def get_train_batch(batch_size):
        nonlocal sampling_list
        if len(sampling_list) < batch_size:
            sample = sampling_list.copy()
            sampling_list = train_data.copy()
            random.shuffle(sampling_list)

            needed = batch_size - len(sample)
            sample = sample + sampling_list[:needed]
            sampling_list = sampling_list[needed:]
        else:
            sample = sampling_list[:batch_size]
            sampling_list = sampling_list[batch_size:]
        sample = [random.sample(x, 2) for x in sample]
        return sample

    max_val_score = evaluate_embedding_model(model, tokenizer, eval_data)

    projection = torch.nn.Linear(model.config.hidden_size, proj_dimension, bias=False).to("cuda")
    torch.nn.init.normal_(projection.weight, mean=0.0, std=1e-1)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projection.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    epochs_without_improvement = -5
    for epoch in range(1, 500):
        model.train()

        for batch_idx in tqdm(range(1000), desc=f"Training (epoch {epoch})", file=sys.stdout):

            next_batch = get_train_batch(32)

            h_1 = [x[0] for x in next_batch]
            h_2 = [x[1] for x in next_batch]

            batch_1 = tokenizer(h_1, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            batch_2 = tokenizer(h_2, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")

            embeddings_1 = projection(model(**batch_1, output_hidden_states=True).hidden_states[-1][:, 0])
            embeddings_2 = projection(model(**batch_2, output_hidden_states=True).hidden_states[-1][:, 0])

            temperature = 0.07

            similarities = torch.nn.functional.cosine_similarity(embeddings_1.unsqueeze(1), embeddings_2.unsqueeze(0), dim=-1)
            logits = similarities / temperature

            labels = torch.arange(logits.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epochs_without_improvement += 1
        val_score = evaluate_embedding_model(model, tokenizer, eval_data)
        if val_score > max_val_score:
            max_val_score = val_score
            torch.save(model.state_dict(), relative_path(f"../data/saved_models/invasion_biology/embedding_model_{prompt_code}.pkl"))
            torch.save(projection.state_dict(), relative_path(f"../data/saved_models/invasion_biology/embedding_proj_{prompt_code}.pkl"))
            print("New best! Model saved.")
            epochs_without_improvement = min(epochs_without_improvement, 0)

        if epochs_without_improvement >= 7:
            break
        print()

def train_all_individual_prompt_embedding_models():
    os.makedirs(relative_path(f"../data/saved_models/invasion_biology/"), exist_ok=True)
    for p in prompt_identifiers:
        if not os.path.exists(relative_path(f"../data/saved_models/invasion_biology/embedding_model_{p}.pkl")):
            print(f"\n\n____________________________________________________________________\nStarting with training of \"{p}\" model!\n")
            train_embedding_model(p)