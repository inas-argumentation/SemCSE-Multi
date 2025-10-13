import itertools
import os
from settings import proj_dimension, encoder_checkpoint
import numpy as np
import torch
import torch.nn.functional as F
from medical_embeddings.auxiliary import relative_path, load_medical_dataset
from transformers import AutoModel, AutoTokenizer
from medical_embeddings.generate_aspect_specific_summaries import prompts

SENTENCE_BATCH_SIZE = 32

dataset = load_medical_dataset()
dataset = list(dataset.values())[:500]

prompt_identifiers = list(prompts.keys())


def load_embedding_model(prompt_code):
    model = AutoModel.from_pretrained(encoder_checkpoint).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    model_path = relative_path(f"../data/saved_models/medical/embedding_model_{prompt_code}.pkl")
    proj_path = relative_path(f"../data/saved_models/medical/embedding_proj_{prompt_code}.pkl")

    if os.path.exists(model_path) and os.path.exists(proj_path):
        model.load_state_dict(torch.load(model_path))
        projection = torch.nn.Linear(model.config.hidden_size, 150, bias=False).to("cuda")
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

    for sample_data in dataset_split:
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
        count = len([x for x in dataset
                     if len([y for y in itertools.chain.from_iterable([x[c] for c in x.keys() if p in c])
                             if 'not applicable' not in y.lower()]) >= 2])
        print(f"{p}: {count}")

    print(f"\n{'=' * 80}")
    print("EMBEDDING MODEL EVALUATION RESULTS")
    print(f"{'=' * 80}")

    main_results = {}

    # Evaluate each embedding model on its own aspect
    print(f"\n{'-' * 50}")
    print("MAIN EVALUATION (Each model on its target aspect)")
    print(f"{'-' * 50}")

    for prompt_code in prompt_identifiers:
        print(f"\nEvaluating embedding model for aspect {prompt_code}:")

        try:
            model, tokenizer, projection = load_embedding_model(prompt_code)
            eval_data = prepare_evaluation_data(dataset, prompt_code)
            print(f"  Number of evaluation samples: {len(eval_data)}")

            avg_rank, mrr = evaluate_retrieval_performance(model, tokenizer, projection, eval_data)
            main_results[prompt_code] = (avg_rank, mrr, len(eval_data))

            print(f"  Average retrieval rank: {avg_rank:.3f}")
            print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")

        except Exception as e:
            print(f"  Error evaluating {prompt_code}: {str(e)}")

    print(f"\n{'-' * 50}")
    print("ABLATION STUDY (Evaluate all models on other aspect's data)")
    print(f"{'-' * 50}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)
    cross_eval = {eval_p: {} for eval_p in prompt_identifiers}

    for eval_prompt in prompt_identifiers:
        eval_data = prepare_evaluation_data(dataset, eval_prompt)

        print(f"\nEvaluating all other models on aspect {eval_prompt} (N={len(eval_data)})")

        for model_prompt in prompt_identifiers:
            model_path = relative_path(f"../data/saved_models/medical/embedding_model_{model_prompt}.pkl")
            proj_path = relative_path(f"../data/saved_models/medical/embedding_proj_{model_prompt}.pkl")
            if not (os.path.exists(model_path) and os.path.exists(proj_path)):
                print(f"  Skipping model {model_prompt}: model/proj file missing.")
                continue

            model = AutoModel.from_pretrained(encoder_checkpoint)
            sd = torch.load(model_path, map_location="cpu")
            model.load_state_dict(sd)
            model.to("cuda")

            projection = torch.nn.Linear(model.config.hidden_size, proj_dimension, bias=False)
            psd = torch.load(proj_path, map_location="cpu")
            projection.load_state_dict(psd)
            projection.to("cuda")

            try:
                avg_rank, mrr = evaluate_retrieval_performance(
                    model, tokenizer, projection, eval_data)
                cross_eval[eval_prompt][model_prompt] = (avg_rank, mrr)
                print(f"  Model {model_prompt:<3} -> avg_rank {avg_rank:.3f}, MRR {mrr:.4f}")
            except Exception as e:
                print(f"  Error evaluating {model_prompt} on {eval_prompt}: {e}")

            del model, projection
            torch.cuda.empty_cache()

        if eval_prompt in cross_eval and eval_prompt in cross_eval[eval_prompt]:
            main_mrr = cross_eval[eval_prompt][eval_prompt][1]
            other_mrrs = [v[1] for k, v in cross_eval[eval_prompt].items() if k != eval_prompt]
            others_mrr_mean = float(np.mean(other_mrrs)) if len(other_mrrs) > 0 else float('nan')
            print(f"      MRR main = {main_mrr:.4f}, others mean = {others_mrr_mean:.4f}")

    # Summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY TABLE (ALL METRICS)")
    print(f"{'=' * 100}")
    header = f"{'Aspect':<8} {'Samples':<8} {'Avg Rank':<12} {'MRR':<8} " \
             f"{'Others Avg Rank':<15} {'Others MRR':<12} "
    print(header)
    print(f"{'-' * 100}")

    for prompt_code in prompt_identifiers:
        if prompt_code in main_results and prompt_code in cross_eval:
            avg_rank, mrr, n_samples = main_results[prompt_code]
            main_mrr = cross_eval[prompt_code][prompt_code][1]

            other_ranks = [v[0] for k, v in cross_eval[prompt_code].items() if k != prompt_code]
            other_mrrs = [v[1] for k, v in cross_eval[prompt_code].items() if k != prompt_code]

            others_avg_rank = np.mean(other_ranks) if other_ranks else float('nan')
            others_mean_mrr = np.mean(other_mrrs) if other_mrrs else float('nan')

            print(
                f"{prompt_code:<8} {n_samples:<8} {avg_rank:<12.3f} {main_mrr:<8.4f} "
                f"{others_avg_rank:<15.3f} {others_mean_mrr:<12.4f} "
            )


def evaluate_baseline_semcse():
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
        eval_data = prepare_evaluation_data(dataset, prompt_code)
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

        similarities = F.cosine_similarity(first_embeddings.unsqueeze(1), second_embeddings.unsqueeze(0), dim=-1)
        ranked_indices = torch.argsort(similarities, dim=-1, descending=True)

        ranks = []
        reciprocal_ranks = []
        for i in range(len(eval_data)):
            rank = (ranked_indices[i] == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))

        avg_rank = np.mean(ranks)
        mrr = np.mean(reciprocal_ranks)

        baseline_results[prompt_code] = (avg_rank, mrr, len(eval_data))

        print(f"  Average retrieval rank: {avg_rank:.3f}")
        print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")

    print(f"\n{'-' * 50}")
    print("BASELINE SUMMARY")
    print(f"{'-' * 50}")
    if baseline_results:
        all_ranks = [v[0] for v in baseline_results.values()]
        all_mrrs = [v[1] for v in baseline_results.values()]
        print(f"Average across aspects - Rank: {np.mean(all_ranks):.3f} ± {np.std(all_ranks):.3f}, "
              f"MRR: {np.mean(all_mrrs):.4f} ± {np.std(all_mrrs):.4f}")

    return baseline_results