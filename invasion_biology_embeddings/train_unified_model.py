import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from invasion_biology_embeddings.auxiliary import relative_path, load_invasion_dataset
import json
from transformers import AutoModel, AutoTokenizer
import pickle
from invasion_biology_embeddings.generate_aspect_specific_summaries import prompts
from invasion_biology_embeddings.models import UnifiedEmbeddingModel
from settings import encoder_checkpoint, proj_dimension

BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LR = 1e-4

prompt_identifiers = list(set([p.split("_")[0] for p in prompts.keys()]))

with open(relative_path("../data/mistral_invasion_biology_summaries.json"), "r") as f:
    dataset = json.load(f)

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


def load_individual_embedding_models():
    models = {}
    projections = {}
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    for p in prompt_identifiers:
        model_path = relative_path(f"../data/saved_models/invasion_biology/embedding_model_{p}.pkl")
        proj_path = relative_path(f"../data/saved_models/invasion_biology/embedding_proj_{p}.pkl")

        if os.path.exists(model_path) and os.path.exists(proj_path):
            model = AutoModel.from_pretrained(encoder_checkpoint).to("cuda")
            model.load_state_dict(torch.load(model_path))
            model.eval()

            projection = torch.nn.Linear(model.config.hidden_size, proj_dimension, bias=False).to("cuda")
            projection.load_state_dict(torch.load(proj_path))
            projection.eval()

            models[p] = model
            projections[p] = projection

    return models, projections, tokenizer


def create_precomputed_embeddings_dataset():
    precomputed_path = relative_path(f"../data/precomputed_embeddings_invasion_biology.pkl")

    if os.path.exists(precomputed_path):
        print("Loading existing precomputed embeddings...")
        with open(precomputed_path, "rb") as f:
            return pickle.load(f)

    print("Creating precomputed embeddings...")
    models, projections, tokenizer = load_individual_embedding_models()

    abstract_dataset = load_invasion_dataset()

    precomputed_data = {}

    for sample_id, sample_data in tqdm(dataset["unlabeled"].items(), desc="Processing samples"):
        abstract = abstract_dataset["unlabeled"][sample_id]["prediction_text"]

        sample_embeddings = {}

        for p in prompt_identifiers:
            summaries = [x for p2 in sample_data for x in sample_data[p2] if p in p2 and "not applicable" not in x.lower()]

            if len(summaries) >= 1:
                with torch.no_grad():
                    inputs = tokenizer(summaries, padding=True, truncation=True,
                                       max_length=510, return_tensors="pt").to("cuda")
                    embeddings = models[p](**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
                    projected_embeddings = projections[p](embeddings)

                    avg_embedding = projected_embeddings.mean(dim=0).cpu().numpy()
                    sample_embeddings[p] = avg_embedding

        if sample_embeddings:
            precomputed_data[sample_id] = {
                "abstract": abstract,
                "embeddings": sample_embeddings
            }

    with open(precomputed_path, "wb") as f:
        pickle.dump(precomputed_data, f)

    print(f"Precomputed embeddings saved for {len(precomputed_data)} samples")
    return precomputed_data


def evaluate_unified_model(model, tokenizer, eval_samples, device, batch_size=32):
    model.eval()

    with torch.no_grad():
        eval_abstracts = [sample["abstract"] for sample in eval_samples]

        all_abstract_embeddings = {key: [] for key in prompt_identifiers}

        for i in range(0, len(eval_abstracts), batch_size):
            batch_abstracts = eval_abstracts[i:i + batch_size]
            inputs = tokenizer(batch_abstracts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            embeddings = model(inputs['input_ids'], inputs['attention_mask'])

            for key in all_abstract_embeddings.keys():
                if key in embeddings:
                    all_abstract_embeddings[key].append(embeddings[key].cpu())

        for key in all_abstract_embeddings.keys():
            if all_abstract_embeddings[key]:
                all_abstract_embeddings[key] = torch.cat(all_abstract_embeddings[key], dim=0).to(device)

        metrics = {}

        for p in prompt_identifiers:
            if p in all_abstract_embeddings and len(all_abstract_embeddings[p]) > 0:
                target_embeddings = []
                valid_indices = []
                for i, sample in enumerate(eval_samples):
                    if p in sample["embeddings"]:
                        target_embeddings.append(sample["embeddings"][p])
                        valid_indices.append(i)

                if len(target_embeddings) > 0:
                    target_embeddings = torch.tensor(np.array(target_embeddings), device=device, dtype=torch.float32)
                    predicted_embeddings = all_abstract_embeddings[p][valid_indices]

                    similarities = batched_similarity(predicted_embeddings, target_embeddings)

                    arg_sorted = torch.argsort(similarities, dim=-1, descending=True)
                    ranks = []
                    for i in range(len(predicted_embeddings)):
                        correct_rank = (arg_sorted[i] == i).nonzero(as_tuple=True)[0].item()
                        ranks.append(correct_rank)

                    metrics[f"{p}_avg_rank"] = np.mean(ranks)

        valid_metrics = [v for v in metrics.values() if v != float('inf')]
        metrics["overall_score"] = np.mean(valid_metrics)
        print(metrics)
    return metrics

def compute_mse_loss(model, tokenizer, batch_abstracts, batch_targets, device):
    inputs = tokenizer(batch_abstracts, padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(device)

    embeddings = model(inputs['input_ids'], inputs['attention_mask'])

    total_loss = 0
    count = 0

    for i, targets in enumerate(batch_targets):
        for p in prompt_identifiers:
            if p in targets and p in embeddings:
                target_embedding = torch.tensor(targets[p], device=device, dtype=torch.float32)
                predicted_embedding = embeddings[p][i]

                loss = 10 * F.mse_loss(predicted_embedding, target_embedding)
                total_loss += loss
                count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0, device=device)

def train_unified_model():
    random.seed(0)
    print("Creating precomputed embeddings dataset...")
    precomputed_data = create_precomputed_embeddings_dataset()

    samples = list(precomputed_data.values())
    random.shuffle(samples)

    train_samples = samples[150:]
    eval_samples = samples[:150]

    sampling_list = train_samples.copy()

    def get_train_batch(batch_size):
        nonlocal sampling_list
        if len(sampling_list) < batch_size:
            sample = sampling_list.copy()
            sampling_list = train_samples.copy()
            random.shuffle(sampling_list)

            needed = batch_size - len(sample)
            sample = sample + sampling_list[:needed]
            sampling_list = sampling_list[needed:]
        else:
            sample = sampling_list[:batch_size]
            sampling_list = sampling_list[batch_size:]
        return sample

    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)
    model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_loss = evaluate_unified_model(model, tokenizer, eval_samples, "cuda", BATCH_SIZE)["overall_score"]
    epochs_without_improvement = 0

    for epoch in range(1, 100):
        model.train()

        for batch_idx in tqdm(range(250), desc=f"Epoch {epoch}"):
            batch_samples = get_train_batch(BATCH_SIZE)

            batch_abstracts = [s["abstract"] for s in batch_samples]
            batch_targets = [s["embeddings"] for s in batch_samples]

            optimizer.zero_grad()

            mse_loss = compute_mse_loss(model, tokenizer, batch_abstracts, batch_targets, "cuda")
            if mse_loss.item() > 0:
                mse_loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)

            optimizer.step()

        eval_loss = evaluate_unified_model(model, tokenizer, eval_samples, "cuda", BATCH_SIZE)["overall_score"]
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), relative_path(f"../data/saved_models/invasion_biology/unified_embedding_model.pkl"))
            print("New best model saved!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 10:
            print("Early stopping.")
            break

    print(f"Training completed. Best loss: {best_loss:.4f}")