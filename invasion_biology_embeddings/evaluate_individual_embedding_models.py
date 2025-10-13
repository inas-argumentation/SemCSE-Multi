import itertools
import os
import numpy as np
import torch
import torch.nn.functional as F
from invasion_biology_embeddings.auxiliary import relative_path
import json
from transformers import AutoModel, AutoTokenizer
from invasion_biology_embeddings.generate_aspect_specific_summaries import prompts
from settings import encoder_checkpoint, proj_dimension

with open(relative_path("../data/mistral_invasion_biology_summaries.json"), "r") as f:
    dataset = json.load(f)

prompt_identifiers = list(set([p.split("_")[0] for p in prompts.keys()]))

SENTENCE_BATCH_SIZE = 32

def load_embedding_model(prompt_code):
    model = AutoModel.from_pretrained(encoder_checkpoint).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    model_path = relative_path(f"../data/saved_models/invasion_biology/embedding_model_{prompt_code}.pkl")
    proj_path = relative_path(f"../data/saved_models/invasion_biology/embedding_proj_{prompt_code}.pkl")

    if os.path.exists(model_path) and os.path.exists(proj_path):
        model.load_state_dict(torch.load(model_path))
        projection = torch.nn.Linear(model.config.hidden_size, proj_dimension, bias=False).to("cuda")
        projection.load_state_dict(torch.load(proj_path))
        return model, tokenizer, projection
    else:
        raise FileNotFoundError(f"Model files not found for {prompt_code}")


def get_embeddings(model, tokenizer, projection, sentences):
    model.eval()
    with torch.no_grad():
        embeddings = []
        batch_size = SENTENCE_BATCH_SIZE

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            batch_embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
            projected_embeddings = projection(batch_embeddings)
            embeddings.append(projected_embeddings)

        return torch.cat(embeddings, dim=0)


def evaluate_retrieval_performance(model, tokenizer, projection, eval_data):
    first_summaries = [sample[0] for sample in eval_data]
    second_summaries = [sample[1] for sample in eval_data]

    first_embeddings = get_embeddings(model, tokenizer, projection, first_summaries)
    second_embeddings = get_embeddings(model, tokenizer, projection, second_summaries)

    similarities = F.cosine_similarity(first_embeddings.unsqueeze(1), second_embeddings.unsqueeze(0), dim=-1)
    ranked_indices = torch.argsort(similarities, dim=-1, descending=True)

    ranks = []
    reciprocal_ranks = []
    for i in range(len(eval_data)):
        # The correct match for query i is at position i in the second_summaries
        rank = (ranked_indices[i] == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
        reciprocal_ranks.append(1 / (rank + 1))

    avg_rank = np.mean(ranks)
    mrr = np.mean(reciprocal_ranks)

    return avg_rank, mrr


def prepare_evaluation_data(dataset_split, prompt_code):
    available_data = []

    for sample_key, sample_data in dataset_split.items():
        summaries = []
        for column_key in sample_data.keys():
            if prompt_code in column_key:
                summaries.extend(sample_data[column_key])

        valid_summaries = [s for s in summaries if "not applicable" not in s.lower()]

        if len(valid_summaries) >= 2:
            available_data.append(valid_summaries[:2])

    return available_data

def main_evaluation():
    print("Analyzing labeled data:")
    for p in prompt_identifiers:
        count = len([x for x in dataset['labeled'].values()
                     if len([y for y in itertools.chain.from_iterable([x[c] for c in x.keys() if p in c])
                             if 'not applicable' not in y.lower()]) >= 2])
        print(f"{p}: {count}")

    print(f"\n{'=' * 80}")
    print("EMBEDDING MODEL EVALUATION RESULTS")
    print(f"{'=' * 80}")

    main_results = {}

    print(f"\n{'-' * 50}")
    print("MAIN EVALUATION (Each model on its target aspect)")
    print(f"{'-' * 50}")

    for prompt_code in prompt_identifiers:
        print(f"\nEvaluating embedding model for aspect {prompt_code}:")

        try:
            model, tokenizer, projection = load_embedding_model(prompt_code)
            eval_data = prepare_evaluation_data(dataset['labeled'], prompt_code)

            if len(eval_data) < 2:
                print(f"  Insufficient data for {prompt_code} (only {len(eval_data)} samples)")
                continue

            print(f"  Number of evaluation samples: {len(eval_data)}")

            avg_rank, mrr = evaluate_retrieval_performance(model, tokenizer, projection, eval_data)

            main_results[prompt_code] = (avg_rank, mrr, len(eval_data))

            print(f"  Average retrieval rank: {avg_rank:.3f}")
            print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")

        except Exception as e:
            print(f"  Error evaluating {prompt_code}: {str(e)}")

    print(f"\n{'-' * 50}")
    print("ABLATION STUDY (Evaluate all models ON each aspect's data)")
    print(f"{'-' * 50}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    cross_eval = {eval_p: {} for eval_p in prompt_identifiers}

    for eval_prompt in prompt_identifiers:
        eval_data = prepare_evaluation_data(dataset['labeled'], eval_prompt)
        if len(eval_data) < 2:
            print(f"Skipping aspect {eval_prompt}: insufficient samples ({len(eval_data)})")
            continue

        print(f"\nEvaluating ALL models on aspect {eval_prompt} (N={len(eval_data)})")

        for model_prompt in prompt_identifiers:
            model_path = relative_path(f"../data/saved_models/invasion_biology/embedding_model_{model_prompt}.pkl")
            proj_path = relative_path(f"../data/saved_models/invasion_biology/embedding_proj_{model_prompt}.pkl")
            if not (os.path.exists(model_path) and os.path.exists(proj_path)):
                print(f"  Skipping model {model_prompt}: model/proj file missing.")
                continue

            model = AutoModel.from_pretrained(encoder_checkpoint)
            sd = torch.load(model_path, map_location="cpu")
            model.load_state_dict(sd)
            model.to("cuda")

            projection = torch.nn.Linear(model.config.hidden_size, 150, bias=False)
            psd = torch.load(proj_path, map_location="cpu")
            projection.load_state_dict(psd)
            projection.to("cuda")

            try:
                avg_rank, mrr = evaluate_retrieval_performance(model, tokenizer, projection, eval_data)
                cross_eval[eval_prompt][model_prompt] = (avg_rank, mrr)
                print(f"  Model {model_prompt:<3} ->  avg_rank {avg_rank:.3f}, MRR {mrr:.4f}")
            except Exception as e:
                print(f"  Error evaluating {model_prompt} on {eval_prompt}: {e}")

            del model, projection
            torch.cuda.empty_cache()

        print(f"\n{'=' * 80}")
        print("ASPECT-SPECIFIC vs. CROSS-ASPECT MODEL PERFORMANCE")
        print(f"{'=' * 80}")

        if cross_eval:
            print(f"\n{'Aspect':<12} {'Matching Model':<20} {'Other Models (Avg)':<20} {'Difference':<15}")
            print(f"{'':12} {'Rank':>9} {'MRR':>9} {'Rank':>9} {'MRR':>9} {'Rank':>7} {'MRR':>7}")
            print("-" * 80)

            all_matching_ranks = []
            all_matching_mrrs = []
            all_other_ranks = []
            all_other_mrrs = []

            for eval_p in sorted(prompt_identifiers):
                if eval_p not in cross_eval or not cross_eval[eval_p]:
                    continue

                matching_rank = None
                matching_mrr = None
                if eval_p in cross_eval[eval_p]:
                    matching_rank, matching_mrr = cross_eval[eval_p][eval_p]

                other_ranks = []
                other_mrrs = []
                for model_p in cross_eval[eval_p]:
                    if model_p != eval_p:
                        avg_rank, mrr = cross_eval[eval_p][model_p]
                        other_ranks.append(avg_rank)
                        other_mrrs.append(mrr)

                avg_other_rank = np.mean(other_ranks) if other_ranks else None
                avg_other_mrr = np.mean(other_mrrs) if other_mrrs else None

                if matching_rank is not None and avg_other_rank is not None:
                    rank_diff = avg_other_rank - matching_rank
                    mrr_diff = matching_mrr - avg_other_mrr

                    print(f"{eval_p:<12} {matching_rank:>9.3f} {matching_mrr:>9.4f} "
                          f"{avg_other_rank:>9.3f} {avg_other_mrr:>9.4f} "
                          f"{rank_diff:>+7.3f} {mrr_diff:>+7.4f}")

                    all_matching_ranks.append(matching_rank)
                    all_matching_mrrs.append(matching_mrr)
                    all_other_ranks.append(avg_other_rank)
                    all_other_mrrs.append(avg_other_mrr)

            print("-" * 80)
            if all_matching_ranks:
                avg_matching_rank = np.mean(all_matching_ranks)
                avg_matching_mrr = np.mean(all_matching_mrrs)
                avg_other_rank = np.mean(all_other_ranks)
                avg_other_mrr = np.mean(all_other_mrrs)

                overall_rank_diff = avg_other_rank - avg_matching_rank
                overall_mrr_diff = avg_matching_mrr - avg_other_mrr

                print(f"{'AVERAGE':<12} {avg_matching_rank:>9.3f} {avg_matching_mrr:>9.4f} "
                      f"{avg_other_rank:>9.3f} {avg_other_mrr:>9.4f} "
                      f"{overall_rank_diff:>+7.3f} {overall_mrr_diff:>+7.4f}")

                print(f"\n{'Summary':}")
                print(
                    f"  Aspect-specific models are better by {overall_rank_diff:.3f} rank positions")
                print(f"  Aspect-specific models are better by {overall_mrr_diff:.4f} MRR points")

        print(f"\n{'=' * 80}\n")


def evaluate_baseline_semcse(metric_type="euclidean"):
    from transformers import AutoModel, AutoTokenizer

    print(f"\n{'-' * 50}")
    print("BASELINE SemCSE EVALUATION")
    print(f"{'-' * 50}")

    print("Loading baseline SemCSE model...")
    baseline_model = AutoModel.from_pretrained("CLAUSE-Bielefeld/SemCSE").to("cuda")
    baseline_tokenizer = AutoTokenizer.from_pretrained("CLAUSE-Bielefeld/SemCSE")
    baseline_model.eval()

    baseline_results = {}

    for prompt_code in prompt_identifiers:
        print(f"\nEvaluating baseline on aspect {prompt_code}:")

        eval_data = prepare_evaluation_data(dataset['labeled'], prompt_code)

        if len(eval_data) < 2:
            print(f"  Insufficient data for {prompt_code} (only {len(eval_data)} samples)")
            continue

        print(f"  Number of evaluation samples: {len(eval_data)}")

        first_summaries = [sample[0] for sample in eval_data]
        second_summaries = [sample[1] for sample in eval_data]

        with torch.no_grad():
            first_embeddings = []
            batch_size = SENTENCE_BATCH_SIZE
            for i in range(0, len(first_summaries), batch_size):
                batch = first_summaries[i:i + batch_size]
                inputs = baseline_tokenizer(batch, padding=True, truncation=True, max_length=128,
                                            return_tensors="pt").to("cuda")
                batch_embeddings = baseline_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
                first_embeddings.append(batch_embeddings)
            first_embeddings = torch.cat(first_embeddings, dim=0)

            second_embeddings = []
            for i in range(0, len(second_summaries), batch_size):
                batch = second_summaries[i:i + batch_size]
                inputs = baseline_tokenizer(batch, padding=True, truncation=True, max_length=128,
                                            return_tensors="pt").to("cuda")
                batch_embeddings = baseline_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
                second_embeddings.append(batch_embeddings)
            second_embeddings = torch.cat(second_embeddings, dim=0)

        if metric_type == "cosine":
            similarities = F.cosine_similarity(first_embeddings.unsqueeze(1), second_embeddings.unsqueeze(0), dim=-1)
        elif metric_type == "euclidean":
            similarities = -torch.cdist(first_embeddings, second_embeddings)
        else:
            raise Exception("Metric not known.")

        ranked_indices = torch.argsort(similarities, dim=-1, descending=True)

        ranks = []
        reciprocal_ranks = []
        for i in range(len(eval_data)):
            rank = (ranked_indices[i] == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))

        avg_rank = np.mean(ranks)
        normalized_score = 1 - (avg_rank / (len(eval_data) - 1))
        mrr = np.mean(reciprocal_ranks)

        baseline_results[prompt_code] = (avg_rank, normalized_score, mrr, len(eval_data))

        print(f"  Average retrieval rank: {avg_rank:.3f}")
        print(f"  Normalized score (1=optimal): {normalized_score:.4f}")
        print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")

    del baseline_model
    torch.cuda.empty_cache()

    print(f"\n{'-' * 50}")
    print("BASELINE SUMMARY")
    print(f"{'-' * 50}")
    if baseline_results:
        all_ranks = [v[0] for v in baseline_results.values()]
        all_scores = [v[1] for v in baseline_results.values()]
        all_mrrs = [v[2] for v in baseline_results.values()]
        print(f"Average across aspects - Rank: {np.mean(all_ranks):.3f} ± {np.std(all_ranks):.3f}, "
              f"Score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}, "
              f"MRR: {np.mean(all_mrrs):.4f} ± {np.std(all_mrrs):.4f}")

    return baseline_results