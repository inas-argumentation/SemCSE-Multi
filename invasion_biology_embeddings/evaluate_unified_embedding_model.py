from scipy.stats import spearmanr
import os
import sys
import json
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from invasion_biology_embeddings.auxiliary import relative_path, load_invasion_dataset
from settings import proj_dimension, encoder_checkpoint
from transformers import AutoTokenizer
from invasion_biology_embeddings.models import UnifiedEmbeddingModel

PAIRS_DIR = relative_path("../data/llm_pairwise_assessments/invasion_biology")
SAVED_MODELS_DIR = relative_path("../data/saved_models")
MISTRAL_SUMMARIES_JSON = relative_path("../data/mistral_invasion_biology_summaries.json")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_aspect_samples_and_texts(aspect_key, pairs_dir=PAIRS_DIR):
    samples_file = os.path.join(pairs_dir, f"{aspect_key}_samples.json")
    if not os.path.exists(samples_file):
        raise FileNotFoundError(f"Samples file not found: {samples_file}")

    with open(samples_file, "r", encoding="utf-8") as f:
        samples_data = json.load(f)

    sample_indices = [int(idx) for idx in samples_data.keys()]
    sample_indices.sort()

    texts_by_index = {}
    for idx_str, sample_info in samples_data.items():
        idx = int(idx_str)
        texts_by_index[idx] = sample_info["abstract"]

    return sample_indices, texts_by_index


def load_pairwise_annotations(aspect_key, pairs_dir=PAIRS_DIR):
    path = os.path.join(pairs_dir, f"{aspect_key}_pairs.jsonl")
    if not os.path.exists(path):
        print(f"[WARN] pairs file not found for {aspect_key}: {path}")
        return {}
    pair_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                a = int(obj["a"])
                b = int(obj["b"])
                score = obj.get("score", None)
                if score is None:
                    continue
                key = (min(a, b), max(a, b))
                pair_map[key] = int(score)
            except Exception as e:
                print(f"[WARN] failed to parse line in {path}: {e}")
                continue
    return pair_map


def load_summarizing_sentences():
    with open(MISTRAL_SUMMARIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries_by_index = {}

    for idx_str, sample_data in data["labeled"].items():
        idx = int(idx_str)
        summaries_by_index[idx] = sample_data

    return summaries_by_index


def gather_aspect_keys():
    keys = []
    for fn in os.listdir(PAIRS_DIR):
        if fn.endswith("_samples.json"):
            keys.append(fn.split("_samples.json")[0])
    keys = sorted(list(set(keys)))
    if not keys:
        raise FileNotFoundError("No aspect sample files found.")
    return keys


def compute_embeddings_for_samples(model, tokenizer, sample_texts, device=DEVICE, batch_size=BATCH_SIZE):
    model.eval()
    per_key_lists = defaultdict(list)
    sample_count = len(sample_texts)

    with torch.no_grad():
        for start in tqdm(range(0, sample_count, batch_size), desc="Computing embeddings", file=sys.stdout):
            batch_texts = sample_texts[start:start + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(enc['input_ids'], enc['attention_mask'])

            for key, tensor in outputs.items():
                arr = tensor.cpu().numpy()
                per_key_lists[key].append(arr)

    embeddings_by_key = {}
    for key, list_of_chunks in per_key_lists.items():
        embeddings_by_key[key] = np.concatenate(list_of_chunks, axis=0)
    return embeddings_by_key


def build_pairwise_similarity_matrix_from_annotations(sample_indices, pair_map):
    N = len(sample_indices)
    similarity_matrix = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            if i != j:
                idx_a = sample_indices[i]
                idx_b = sample_indices[j]
                key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if key in pair_map:
                    similarity_matrix[i, j] = float(pair_map[key])
                else:
                    raise Exception("Missing key!")

    return similarity_matrix


def build_label_sharing_matrix(sample_indices, labels_by_index):
    N = len(sample_indices)
    label_matrix = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            if i != j:
                idx_a = sample_indices[i]
                idx_b = sample_indices[j]
                labels_a = labels_by_index.get(str(idx_a), set())
                labels_b = labels_by_index.get(str(idx_b), set())
                if len(labels_a & labels_b) > 0:
                    label_matrix[i, j] = 1.0

    return label_matrix


def get_top_k_with_ties(scores, k=10):
    if len(scores) <= k:
        return set(range(len(scores)))

    sorted_indices = np.argsort(scores)[::-1]
    kth_score = scores[sorted_indices[k - 1]]

    top_k_indices = set()
    for idx in range(len(scores)):
        if scores[idx] >= kth_score:
            top_k_indices.add(idx)

    return top_k_indices


def compute_top_k_accuracy_matrix(pred_similarity_matrix, gt_relevance_matrix, k=10):
    N = pred_similarity_matrix.shape[0]
    sample_accuracies = []

    for i in range(N):
        pred_scores = []
        gt_scores = []
        valid_indices = []

        for j in range(N):
            if i != j:
                pred_scores.append(pred_similarity_matrix[i, j])
                gt_scores.append(gt_relevance_matrix[i, j])
                valid_indices.append(j)

        if len(pred_scores) == 0:
            continue

        pred_scores = np.array(pred_scores)
        gt_scores = np.array(gt_scores)

        pred_top_k_local = get_top_k_with_ties(pred_scores, k)
        gt_top_k_local = get_top_k_with_ties(gt_scores, k)

        pred_top_k = set(valid_indices[idx] for idx in pred_top_k_local)
        gt_top_k = set(valid_indices[idx] for idx in gt_top_k_local)

        if len(pred_top_k) > 0:
            accuracy = len(pred_top_k & gt_top_k) / len(pred_top_k)
            sample_accuracies.append(accuracy)

    if len(sample_accuracies) == 0:
        return None

    return float(np.mean(sample_accuracies))


def compute_mean_spearman_correlation(pred_similarity_matrix, gt_relevance_matrix):

    N = pred_similarity_matrix.shape[0]
    sample_correlations = []

    for i in range(N):
        predictions = []
        relevances = []
        other_indices = []

        for j in range(N):
            if i != j:
                predictions.append(pred_similarity_matrix[i, j])
                relevances.append(gt_relevance_matrix[i, j])
                other_indices.append(j)

        if not other_indices:
            continue

        try:
            pred_array = np.array(predictions)
            rel_array = np.array(relevances)

            if len(np.unique(pred_array)) > 1 and len(np.unique(rel_array)) > 1:
                r = spearmanr(pred_array, rel_array).correlation
                if not np.isnan(r):
                    sample_correlations.append(r)
        except Exception:
            continue

    if len(sample_correlations) == 0:
        return None

    return float(np.mean(sample_correlations))


def print_results_table(results_matrix, title, pred_aspects=None):
    aspects = ["hypo_labels", "hypothesis", "ecosystem", "researchquestion", "species", "methodology", "recommendation"]
    if pred_aspects is None:
        pred_aspects = ["hypo_labels", "hypothesis", "ecosystem", "researchquestion", "species", "methodology", "recommendation"]

    print(f"\n=== {title} ===")

    header = ["pred\\gt"] + aspects
    col_widths = [max(8, len(header[0]))] + [max(8, len(asp)) for asp in aspects]

    for pred_asp in pred_aspects:
        col_widths[0] = max(col_widths[0], len(pred_asp))
        for i, gt_asp in enumerate(aspects, 1):
            val = results_matrix.get(pred_asp, {}).get(gt_asp, None)
            val_str = "NA" if val is None else f"{val:.4f}"
            col_widths[i] = max(col_widths[i], len(val_str))

    header_row = []
    for i, col in enumerate(header):
        header_row.append(col.ljust(col_widths[i]))
    print(" | ".join(header_row))

    separator = []
    for width in col_widths:
        separator.append("-" * width)
    print("-+-".join(separator))

    for pred_asp in pred_aspects:
        row = [pred_asp.ljust(col_widths[0])]
        for i, gt_asp in enumerate(aspects, 1):
            val = results_matrix.get(pred_asp, {}).get(gt_asp, None)
            val_str = "NA" if val is None else f"{val:.4f}"
            row.append(val_str.ljust(col_widths[i]))
        print(" | ".join(row))


def main():
    print(f"[INFO] Device: {DEVICE}")

    print("[INFO] Loading summarizing sentences dataset...")
    summaries_by_index = load_summarizing_sentences()
    print(f"[INFO] Loaded summaries for {len(summaries_by_index)} samples")

    aspects = gather_aspect_keys()
    print(f"[INFO] Found aspects: {aspects}")

    print(f"[INFO] Loading unified model")
    model = UnifiedEmbeddingModel(encoder_checkpoint, aspects, proj_dimension).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    try:
        state = torch.load(relative_path("../data/saved_models/invasion_biology/unified_embedding_model.pkl"), map_location=DEVICE)
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Failed to load model state: {e}")

    abstract_dataset = load_invasion_dataset()
    labels_by_index = {}
    for idx, sample in abstract_dataset["labeled"].items():
        labels_by_index[idx] = set(sample.get("labels", set()))

    full_indices = sorted(labels_by_index.keys())
    print(f"[INFO] Preparing full-dataset label evaluation on {len(full_indices)} samples...")
    full_texts_by_index = {idx: sample.get("prediction_text", "") for idx, sample in abstract_dataset["labeled"].items()}
    full_texts = [full_texts_by_index[idx] for idx in full_indices]
    full_label_sharing_matrix = build_label_sharing_matrix(full_indices, labels_by_index)

    print("[INFO] Computing full-dataset embeddings for label-based evaluation (this may take a while)...")
    full_embeddings_by_aspect = compute_embeddings_for_samples(model, tokenizer, full_texts, DEVICE, BATCH_SIZE)
    print(f"[INFO] Computed full-dataset embeddings for aspects: {list(full_embeddings_by_aspect.keys())}")

    aggregated_label_spearman = {}
    aggregated_label_top10 = {}
    for pred_aspect, emb in full_embeddings_by_aspect.items():
        try:
            pred_similarity_full = cosine_similarity(emb, emb)
            spearman_val = compute_mean_spearman_correlation(pred_similarity_full, full_label_sharing_matrix)
            top10_val = compute_top_k_accuracy_matrix(pred_similarity_full, full_label_sharing_matrix, k=10)
        except Exception as e:
            print(f"[WARN] Failed full-dataset label eval for {pred_aspect}: {e}")
            spearman_val = None
            top10_val = None

        aggregated_label_spearman[pred_aspect] = spearman_val
        aggregated_label_top10[pred_aspect] = top10_val

    results_spearman = {asp: {} for asp in aspects}
    results_top10 = {asp: {} for asp in aspects}

    results_spearman['hypo_labels'] = {}
    results_top10['hypo_labels'] = {}

    for gt_aspect in aspects:
        print(f"\n[PROCESSING] Ground truth aspect: {gt_aspect}")

        try:
            sample_indices, texts_by_index = load_aspect_samples_and_texts(gt_aspect, PAIRS_DIR)
            print(f"[INFO] Loaded {len(sample_indices)} samples for aspect {gt_aspect}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

        pair_map = load_pairwise_annotations(gt_aspect, PAIRS_DIR)
        print(f"[INFO] Loaded {len(pair_map)} pairwise annotations for aspect {gt_aspect}")

        if len(pair_map) == 0:
            print(f"[WARN] No pairwise annotations found for {gt_aspect}, skipping")
            continue

        sample_texts = [texts_by_index[idx] for idx in sample_indices]

        embeddings_by_aspect = compute_embeddings_for_samples(model, tokenizer, sample_texts, DEVICE, BATCH_SIZE)
        print(f"[INFO] Computed embeddings for aspects: {list(embeddings_by_aspect.keys())}")

        gt_similarity_matrix = build_pairwise_similarity_matrix_from_annotations(sample_indices, pair_map)
        label_sharing_matrix = build_label_sharing_matrix(sample_indices, labels_by_index)

        label_spearman = compute_mean_spearman_correlation(label_sharing_matrix, gt_similarity_matrix)
        results_spearman['hypo_labels'][gt_aspect] = label_spearman
        print(f"[RESULT] pred=hypo_labels vs gt={gt_aspect} -> Spearman = {label_spearman}")

        label_top10 = compute_top_k_accuracy_matrix(label_sharing_matrix, gt_similarity_matrix, k=10)
        results_top10['hypo_labels'][gt_aspect] = label_top10
        print(f"[RESULT] pred=hypo_labels vs gt={gt_aspect} -> Top-10 Acc = {label_top10}")

        for pred_aspect in embeddings_by_aspect.keys():
            pred_embeddings = embeddings_by_aspect[pred_aspect]
            pred_similarity_matrix = cosine_similarity(pred_embeddings, pred_embeddings)

            spearman_corr = compute_mean_spearman_correlation(pred_similarity_matrix, gt_similarity_matrix)
            results_spearman[pred_aspect][gt_aspect] = spearman_corr
            print(f"[RESULT] pred={pred_aspect} vs gt={gt_aspect} -> Spearman = {spearman_corr}")

            top10_acc = compute_top_k_accuracy_matrix(pred_similarity_matrix, gt_similarity_matrix, k=10)
            results_top10[pred_aspect][gt_aspect] = top10_acc
            print(f"[RESULT] pred={pred_aspect} vs gt={gt_aspect} -> Top-10 Acc = {top10_acc}")

    for pred_aspect in aspects:
        results_spearman[pred_aspect]['hypo_labels'] = aggregated_label_spearman[pred_aspect]
        results_top10[pred_aspect]['hypo_labels'] = aggregated_label_top10[pred_aspect]

    results_spearman['hypo_labels']['hypo_labels'] = 1.0
    results_top10['hypo_labels']['hypo_labels'] = 1.0

    print_results_table(results_spearman, "Mean Spearman Correlation")
    print_results_table(results_top10, "Top-10 Accuracy")

    print("\nEvaluation completed.")

def evaluate_baseline_models():
    from transformers import AutoTokenizer, AutoModel
    try:
        from adapters import AutoAdapterModel
    except Exception:
        AutoAdapterModel = None

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import pairwise_distances

    print(f"[INFO] Device: {DEVICE}")
    print("[INFO] Loading summarizing sentences dataset...")
    summaries_by_index = load_summarizing_sentences()
    print(f"[INFO] Loaded summaries for {len(summaries_by_index)} samples")

    aspects = gather_aspect_keys()
    print(f"[INFO] Found aspects: {aspects}")

    abstract_dataset = load_invasion_dataset()
    labels_by_index = {}
    for idx, sample in abstract_dataset["labeled"].items():
        labels_by_index[idx] = set(sample.get("labels", set()))

    full_indices = sorted(labels_by_index.keys())
    print(f"[INFO] Preparing full-dataset label evaluation on {len(full_indices)} samples...")
    full_texts_by_index = {idx: sample.get('prediction_text', '') for idx, sample in abstract_dataset["labeled"].items()}
    full_texts = [full_texts_by_index[idx] for idx in full_indices]
    full_label_sharing_matrix = build_label_sharing_matrix(full_indices, labels_by_index)

    # Define baseline models to evaluate
    baseline_defs = [
        {"name": "OntoInvasion", "hf_id": "CLAUSE-Bielefeld/InvDef-DeBERTa", "loader": "transformers", "distance": "euclidean"},
        {"name": "SemCSE", "hf_id": "CLAUSE-Bielefeld/SemCSE", "loader": "transformers", "distance": "euclidean"},
        {"name": "SPECTER", "hf_id": "allenai/specter", "loader": "transformers", "distance": "euclidean"},
        {"name": "SPECTER2", "hf_id": "allenai/specter2_base", "loader": "adapter", "distance": "euclidean", "adapter": "allenai/specter2"},
        {"name": "SciNCL", "hf_id": "malteos/scincl", "loader": "transformers", "distance": "cosine"},
    ]

    results_spearman = {b["name"]: {} for b in baseline_defs}
    results_top10 = {b["name"]: {} for b in baseline_defs}

    aggregated_label_spearman = {}
    aggregated_label_top10 = {}

    def _compute_embeddings_for_baseline(hf_id, texts, loader="transformers", adapter_name=None, custom_weights=None):
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if loader == "adapter":
            if AutoAdapterModel is None:
                raise RuntimeError("adapters library not available, cannot load adapter models (SPECTER2).")
            model = AutoAdapterModel.from_pretrained(hf_id)
            if adapter_name is not None:
                model.load_adapter(adapter_name, source="hf", set_active=True)
        else:
            model = AutoModel.from_pretrained(hf_id)
            if custom_weights is not None:
                model.load_state_dict(torch.load(custom_weights))
            print("Weights loaded!")

        model.to(DEVICE)
        model.eval()

        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[start:start + BATCH_SIZE]
                enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                outs = model(**enc)

                # use first token ([CLS] / first token) as embedding
                last_h = outs.last_hidden_state
                emb = last_h[:, 0, :].detach().cpu().numpy()
                chunks.append(emb)

        if not chunks:
            return np.zeros((0, 0))
        return np.concatenate(chunks, axis=0)

    for bdef in baseline_defs:
        name = bdef["name"]
        hf_id = bdef["hf_id"]
        loader = bdef.get("loader", "transformers")
        adapter_name = bdef.get("adapter", None)
        distance_type = bdef.get("distance", "cosine")
        custom_weights = bdef.get("custom_weights", None)

        print(f"\n[BASELINE-FULL] {name} - extracting full-dataset embeddings for label eval (hf id: {hf_id})")
        try:
            embeddings_full = _compute_embeddings_for_baseline(hf_id, full_texts, loader=loader, adapter_name=adapter_name, custom_weights=custom_weights)
            if distance_type == "cosine":
                pred_similarity_full = cosine_similarity(embeddings_full, embeddings_full)
            elif distance_type == "euclidean":
                dists = pairwise_distances(embeddings_full, metric="euclidean")
                pred_similarity_full = -dists
            else:
                pred_similarity_full = cosine_similarity(embeddings_full, embeddings_full)

            label_spearman_full = compute_mean_spearman_correlation(pred_similarity_full, full_label_sharing_matrix)
            label_top10_full = compute_top_k_accuracy_matrix(pred_similarity_full, full_label_sharing_matrix, k=10)

            print(f"[RESULT-FULL] baseline={name} vs full_labels -> Spearman = {label_spearman_full}")
            print(f"[RESULT-FULL] baseline={name} vs full_labels -> Top-10 Acc = {label_top10_full}")
        except Exception as e:
            print(f"[WARN] Failed to run full-dataset label eval for baseline {name}: {e}")
            label_spearman_full = None
            label_top10_full = None

        aggregated_label_spearman[name] = label_spearman_full
        aggregated_label_top10[name] = label_top10_full

    for gt_aspect in aspects:
        print(f"\n[PROCESSING] Ground truth aspect: {gt_aspect}")

        try:
            sample_indices, texts_by_index = load_aspect_samples_and_texts(gt_aspect, PAIRS_DIR)
            print(f"[INFO] Loaded {len(sample_indices)} samples for aspect {gt_aspect}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

        pair_map = load_pairwise_annotations(gt_aspect, PAIRS_DIR)
        print(f"[INFO] Loaded {len(pair_map)} pairwise annotations for aspect {gt_aspect}")

        if len(pair_map) == 0:
            print(f"[WARN] No pairwise annotations found for {gt_aspect}, skipping")
            continue

        sample_texts = [texts_by_index[idx] for idx in sample_indices]
        gt_similarity_matrix = build_pairwise_similarity_matrix_from_annotations(sample_indices, pair_map)

        for bdef in baseline_defs:
            name = bdef["name"]
            hf_id = bdef["hf_id"]
            loader = bdef.get("loader", "transformers")
            adapter_name = bdef.get("adapter", None)
            distance_type = bdef.get("distance", "cosine")
            custom_weights = bdef.get("custom_weights", None)

            print(f"\n[BASELINE] {name} - extracting embeddings (hf id: {hf_id})")
            try:
                embeddings = _compute_embeddings_for_baseline(hf_id, sample_texts, loader=loader, adapter_name=adapter_name, custom_weights=custom_weights)
            except Exception as e:
                print(f"[WARN] Failed to run baseline {name}: {e}")
                results_spearman[name][gt_aspect] = None
                results_top10[name][gt_aspect] = None
                continue

            try:
                if distance_type == "cosine":
                    pred_similarity_matrix = cosine_similarity(embeddings, embeddings)
                elif distance_type == "euclidean":
                    dists = pairwise_distances(embeddings, metric="euclidean")
                    pred_similarity_matrix = -dists
                else:
                    pred_similarity_matrix = cosine_similarity(embeddings, embeddings)
            except Exception as e:
                print(f"[WARN] Failed to compute similarity matrix for {name}: {e}")
                results_spearman[name][gt_aspect] = None
                results_top10[name][gt_aspect] = None
                continue

            try:
                spearman_corr = compute_mean_spearman_correlation(pred_similarity_matrix, gt_similarity_matrix)
                top10_acc = compute_top_k_accuracy_matrix(pred_similarity_matrix, gt_similarity_matrix, k=10)

                results_spearman[name][gt_aspect] = spearman_corr
                results_top10[name][gt_aspect] = top10_acc

                print(f"[RESULT] baseline={name} vs gt={gt_aspect} -> Spearman = {spearman_corr}")
                print(f"[RESULT] baseline={name} vs gt={gt_aspect} -> Top-10 Acc = {top10_acc}")
            except Exception as e:
                print(f"[WARN] Failed to evaluate baseline {name} vs gt: {e}")
                results_spearman[name][gt_aspect] = None
                results_top10[name][gt_aspect] = None

    for bdef in baseline_defs:
        name = bdef["name"]
        results_spearman[name]["hypo_labels"] = aggregated_label_spearman.get(name, None)
        results_top10[name]["hypo_labels"] = aggregated_label_top10.get(name, None)

    print_results_table(results_spearman, "Baseline Models - Mean Spearman Correlation", pred_aspects=results_spearman.keys())
    print_results_table(results_top10, "Baseline Models - Top-10 Accuracy", pred_aspects=results_top10.keys())

    print("\nBaseline evaluation completed.")
    return results_spearman, results_top10

if __name__ == "__main__":
    main()