import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, \
    BitsAndBytesConfig
from invasion_biology_embeddings.auxiliary import relative_path, load_invasion_dataset
from invasion_biology_embeddings.models import UnifiedEmbeddingModel, LlamaTransformationModel, MistralTransformationModel
from settings import encoder_checkpoint, proj_dimension
from invasion_biology_embeddings.generate_aspect_specific_summaries import prompts
from t_SNE_utils.t_SNE_processing_functions import calculate_t_SNE_embeddings, embed_new_point, optimize_embedding_for_t_SNE

prompt_identifiers = list(set([p.split("_")[0] for p in prompts.keys()]))

LLAMA_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
MISTRAL_CHECKPOINT = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

UNIFIED_MODEL_PATH = relative_path(f"../data/saved_models/invasion_biology/unified_embedding_model.pkl")
LLAMA_DECODER_DIR = relative_path("../data/saved_models/invasion_biology/llama_decoders")
MISTRAL_DECODER_DIR = relative_path("../data/saved_models/invasion_biology/mistral_decoders")
DATASET_PATH = relative_path("../data/mistral_invasion_biology_summaries.json")

K_TOKENS = 5
TOTAL_NUM_ADDED_TOKENS = 10

def _normalize_sample_embedding(embedding):
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding)
    if embedding.ndim == 1:
        return embedding.unsqueeze(0)
    if embedding.ndim == 2 and embedding.size(0) == 1:
        return embedding
    return embedding.mean(dim=0, keepdim=True)

def _collect_samples_by_type(unified_model, unified_tokenizer, eval_data, embedding_types, device):
    samples_by_type = {etype: [] for etype in embedding_types}
    with torch.no_grad():
        for sample_id, sample_data in tqdm(eval_data.items(), desc="Collecting embeddings"):
            abstract = sample_data["abstract"].strip()
            inputs = unified_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512).to(device)
            unified_embeddings = unified_model(inputs['input_ids'], inputs['attention_mask'])

            for etype in embedding_types:
                summaries = sample_data.get(etype, [])

                valid_summaries = [s for s in summaries if s and "not applicable" not in s.lower()]
                if not valid_summaries:
                    continue

                embedding = unified_embeddings[etype.split("_")[0]]
                embedding = _normalize_sample_embedding(embedding).to(device)
                samples_by_type[etype].append((sample_id, embedding, valid_summaries))
    return samples_by_type


def calculate_llama_reconstruction_loss(llama_model, tokenizer, transform_model, input_embeddings, target_texts):
    transformed_embeddings = transform_model(input_embeddings)
    target_encoding = tokenizer(target_texts, truncation=True, max_length=256, return_tensors="pt", padding=True).to(
        transformed_embeddings.device)

    target_embeddings = llama_model.get_input_embeddings()(target_encoding["input_ids"])

    concat_embeddings = torch.cat([
        target_embeddings[:, 0].unsqueeze(1),
        transformed_embeddings,
        target_embeddings[:, 1:]
    ], dim=1)

    labels = torch.cat([torch.full((len(target_texts), transformed_embeddings.shape[1] + 1), -100, dtype=torch.long,
                                   device=transformed_embeddings.device),
                        target_encoding["input_ids"][:, 1:]], dim=1)

    outputs = llama_model(inputs_embeds=concat_embeddings, labels=labels)
    return outputs.loss


def calculate_mistral_reconstruction_loss(llm, processor, decoder, embedding_batch, target_texts, device):
    llm_dtype = next(llm.parameters()).dtype
    decoder_output_embeddings = decoder(embedding_batch.to(device), llm_dtype)

    placeholder_user_content = ''.join([processor.tokenizer.pad_token] * TOTAL_NUM_ADDED_TOKENS)

    conversation = [{"role": "system", "content": ""},
                    {"role": "user", "content": placeholder_user_content},
                    {"role": "assistant", "content": target_texts[0][:1200]}]
    full_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    full_encoding = processor(text=[full_text], truncation=True, max_length=512,
                              return_tensors="pt", padding=True, add_special_tokens=False, padding_side="right").to(
        device)

    full_input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]
    if not torch.all(attention_mask.eq(torch.tensor(1, device=device))):
        raise Exception("Attention mask!")

    pad_id = processor.tokenizer.pad_token_id
    seq_to_match = torch.tensor([pad_id] * TOTAL_NUM_ADDED_TOKENS, device=device)

    placeholder_start = -1
    for i in range(full_input_ids.size(1) - TOTAL_NUM_ADDED_TOKENS + 1):
        window = full_input_ids[0, i:i + TOTAL_NUM_ADDED_TOKENS]
        if torch.equal(window, seq_to_match):
            placeholder_start = i
            break

    if placeholder_start == -1:
        raise Exception("Error")

    full_embeddings = llm.get_input_embeddings()(full_input_ids)
    inputs_embeds = full_embeddings.clone()
    inputs_embeds[0, placeholder_start:placeholder_start + TOTAL_NUM_ADDED_TOKENS] = decoder_output_embeddings[0]

    labels = full_input_ids.clone()
    labels[0, :placeholder_start + TOTAL_NUM_ADDED_TOKENS + 1] = -100
    labels[labels.eq(processor.tokenizer.pad_token_id)] = -100
    labels[labels.eq(processor.tokenizer.eos_token_id)] = -100

    outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
    return outputs.loss


def load_evaluation_data():
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r") as f:
        all_data = json.load(f)

    abstract_dataset = load_invasion_dataset()
    for idx in abstract_dataset["labeled"]:
        all_data["labeled"][idx]["abstract"] = abstract_dataset["labeled"][idx]["prediction_text"]
    return all_data["labeled"]


def evaluate_all_llama_decoders():
    print("\n--- Evaluating Llama Decoders ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Llama model...")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_CHECKPOINT, torch_dtype=torch.bfloat16,
                                                       device_map="auto").eval()

    print("Loading Unified Embedding model...")
    unified_model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to(device).eval()
    unified_model.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=device))
    unified_tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    eval_data = load_evaluation_data()
    embedding_types = list(prompts.keys())
    results = {etype: {'losses': [], 'perplexities': []} for etype in embedding_types}

    transform_models = {}
    for embedding_type in embedding_types:
        decoder_path = os.path.join(LLAMA_DECODER_DIR, f"llama_decoder_{embedding_type}.pth")

        transform_model = LlamaTransformationModel(
            embedding_dim=proj_dimension,
            llama_hidden_size=llama_model.config.hidden_size,
            model_dtype=llama_model.dtype,
            decoded_tokens=K_TOKENS
        ).to(device).eval()
        transform_model.load_state_dict(torch.load(decoder_path, map_location=device))
        transform_models[embedding_type] = transform_model

    with torch.no_grad():
        for sample_id, sample_data in tqdm(eval_data.items(), desc="Evaluating Llama Samples"):
            abstract = sample_data["abstract"].strip()

            inputs = unified_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512).to(device)
            unified_embeddings = unified_model(inputs['input_ids'], inputs['attention_mask'])

            for etype in embedding_types:
                summaries = sample_data.get(etype, [])
                valid_summaries = [s for s in summaries if s and "not applicable" not in s.lower()]
                if not valid_summaries:
                    continue

                embedding = unified_embeddings[etype.split("_")[0]]
                sample_losses = []
                sample_perplexities = []
                for summary in valid_summaries:
                    loss = calculate_llama_reconstruction_loss(llama_model, llama_tokenizer, transform_models[etype],
                                                               embedding, [summary])
                    if not torch.isnan(loss):
                        loss_val = loss.item()
                        perplexity = torch.exp(loss).item()
                        sample_losses.append(loss_val)
                        sample_perplexities.append(perplexity)
                    else:
                        raise Exception("Loss error")

                if sample_losses:
                    results[etype]['losses'].append(np.mean(sample_losses))
                    results[etype]['perplexities'].append(np.mean(sample_perplexities))

    print("\n--- Llama Decoder Results ---")
    for etype, metrics in results.items():
        if metrics['losses']:
            avg_loss = np.mean(metrics['losses'])
            avg_perplexity = np.mean(metrics['perplexities'])
            print(
                f"  - {etype}: Average Loss = {avg_loss:.4f}, Average Perplexity = {avg_perplexity:.4f} (over {len(metrics['losses'])} samples)")
    print("-" * 40)


def evaluate_all_mistral_decoders():
    print("\n--- Evaluating Mistral Decoders ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Mistral model...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    mistral_model = AutoModelForImageTextToText.from_pretrained(MISTRAL_CHECKPOINT, quantization_config=quant_config,
                                                                torch_dtype=torch.bfloat16, device_map="auto",
                                                                trust_remote_code=True).eval()
    mistral_processor = AutoProcessor.from_pretrained(MISTRAL_CHECKPOINT, trust_remote_code=True)
    if mistral_processor.tokenizer.pad_token is None:
        mistral_processor.tokenizer.pad_token = mistral_processor.tokenizer.eos_token

    print("Loading Unified Embedding model...")
    unified_model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to(device).eval()
    unified_model.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=device))
    unified_tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    eval_data = load_evaluation_data()
    embedding_types = list(prompts.keys())
    results = {etype: {'losses': [], 'perplexities': []} for etype in embedding_types}

    transform_models = {}
    for embedding_type in embedding_types:
        decoder_path = os.path.join(MISTRAL_DECODER_DIR, f"decoder_{embedding_type}.pth")

        transform_model = MistralTransformationModel(
            embedding_dim=proj_dimension,
            llm_hidden_size=mistral_model.config.text_config.hidden_size,
            num_prompt_tokens=K_TOKENS,
            num_mapped_tokens=K_TOKENS
        ).to(device).eval()
        transform_model.load_state_dict(torch.load(decoder_path, map_location=device))
        transform_models[embedding_type] = transform_model

    with torch.no_grad():
        for sample_id, sample_data in tqdm(eval_data.items(), desc="Evaluating Mistral Samples"):
            abstract = sample_data["abstract"].strip()

            inputs = unified_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512).to(device)
            unified_embeddings = unified_model(inputs['input_ids'], inputs['attention_mask'])

            for etype in embedding_types:
                summaries = sample_data.get(etype, [])

                valid_summaries = [s for s in summaries if s and "not applicable" not in s.lower()]
                if not valid_summaries:
                    continue

                embedding = unified_embeddings[etype.split("_")[0]]
                sample_losses = []
                sample_perplexities = []
                for summary in valid_summaries:
                    loss = calculate_mistral_reconstruction_loss(mistral_model, mistral_processor,
                                                                 transform_models[etype], embedding, [summary], device)
                    if not torch.isnan(loss):
                        loss_val = loss.item()
                        perplexity = torch.exp(loss).item()
                        sample_losses.append(loss_val)
                        sample_perplexities.append(perplexity)

                if sample_losses:
                    results[etype]['losses'].append(np.mean(sample_losses))
                    results[etype]['perplexities'].append(np.mean(sample_perplexities))

    print("\n--- Mistral Decoder Results ---")
    for etype, metrics in results.items():
        if metrics['losses']:
            avg_loss = np.mean(metrics['losses'])
            avg_perplexity = np.mean(metrics['perplexities'])
            print(
                f"  - {etype}: Average Loss = {avg_loss:.4f}, Average Perplexity = {avg_perplexity:.4f} (over {len(metrics['losses'])} samples)")
    print("-" * 42)

# Ablation: Try to reconstruct mismatched summary-embedding pair
def evaluate_llama_decoders_shuffled():
    print("\n--- Evaluating Llama Decoders (Shuffled Ablation) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Llama model...")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_CHECKPOINT, torch_dtype=torch.bfloat16,
                                                       device_map="auto").eval()

    print("Loading Unified Embedding model...")
    unified_model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to(device).eval()
    unified_model.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=device))
    unified_tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    eval_data = load_evaluation_data()
    embedding_types = list(prompts.keys())

    transform_models = {}
    for embedding_type in embedding_types:
        decoder_path = os.path.join(LLAMA_DECODER_DIR, f"llama_decoder_{embedding_type}.pth")
        transform_model = LlamaTransformationModel(
            embedding_dim=proj_dimension,
            llama_hidden_size=llama_model.config.hidden_size,
            model_dtype=llama_model.dtype,
            decoded_tokens=K_TOKENS
        ).to(device).eval()
        transform_model.load_state_dict(torch.load(decoder_path, map_location=device))
        transform_models[embedding_type] = transform_model

    print("Preparing embeddings and summaries for shuffling...")
    embeddings_by_type = {etype: [] for etype in embedding_types}
    summaries_by_type = {etype: [] for etype in embedding_types}

    with torch.no_grad():
        for sample_id, sample_data in tqdm(eval_data.items(), desc="Extracting embeddings and summaries"):
            abstract = sample_data["abstract"].strip()
            inputs = unified_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512).to(device)
            unified_embeddings = unified_model(inputs['input_ids'], inputs['attention_mask'])

            for etype in embedding_types:
                summaries = sample_data.get(etype, [])

                valid_summaries = [s for s in summaries if s and "not applicable" not in s.lower()]
                if not valid_summaries:
                    continue

                embedding = unified_embeddings[etype.split("_")[0]]

                embeddings_by_type[etype].append(embedding)
                summaries_by_type[etype].append(valid_summaries)

    print("Shuffling embedding-summary pairs...")
    results = {etype: {'losses': [], 'perplexities': []} for etype in embedding_types}

    for etype in embedding_types:
        if not embeddings_by_type[etype]:
            continue

        shuffled_indices = list(range(len(summaries_by_type[etype])))
        random.shuffle(shuffled_indices)
        shuffled_summary_lists = [summaries_by_type[etype][i] for i in shuffled_indices]

        print(f"Evaluating shuffled {etype} embeddings...")
        with torch.no_grad():
            for embedding, shuffled_summary_list in tqdm(zip(embeddings_by_type[etype], shuffled_summary_lists), desc=f"Evaluating aspect {etype}"):
                sample_losses = []
                sample_perplexities = []
                for summary in shuffled_summary_list:
                    loss = calculate_llama_reconstruction_loss(
                        llama_model, llama_tokenizer, transform_models[etype],
                        embedding, [summary])
                    if not torch.isnan(loss):
                        loss_val = loss.item()
                        perplexity = torch.exp(loss).item()
                        sample_losses.append(loss_val)
                        sample_perplexities.append(perplexity)

                if sample_losses:
                    results[etype]['losses'].append(np.mean(sample_losses))
                    results[etype]['perplexities'].append(np.mean(sample_perplexities))

    print("\n--- Llama Decoder Shuffled Results ---")
    for etype, metrics in results.items():
        if metrics['losses']:
            avg_loss = np.mean(metrics['losses'])
            avg_perplexity = np.mean(metrics['perplexities'])
            print(
                f"  - {etype}: Average Loss = {avg_loss:.4f}, Average Perplexity = {avg_perplexity:.4f} (over {len(metrics['losses'])} samples)")
    print("-" * 47)

# Ablation: Try to reconstruct mismatched summary-embedding pair
def evaluate_mistral_decoders_shuffled():
    print("\n--- Evaluating Mistral Decoders (Shuffled Ablation) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Mistral model...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    mistral_model = AutoModelForImageTextToText.from_pretrained(
        MISTRAL_CHECKPOINT, quantization_config=quant_config, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    ).eval()
    mistral_processor = AutoProcessor.from_pretrained(MISTRAL_CHECKPOINT, trust_remote_code=True)
    if mistral_processor.tokenizer.pad_token is None:
        mistral_processor.tokenizer.pad_token = mistral_processor.tokenizer.eos_token

    print("Loading Unified Embedding model...")
    unified_model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to(device).eval()
    unified_model.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=device))
    unified_tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    eval_data = load_evaluation_data()
    embedding_types = list(prompts.keys())

    transform_models = {}
    for embedding_type in embedding_types:
        decoder_path = os.path.join(MISTRAL_DECODER_DIR, f"decoder_{embedding_type}.pth")
        transform_model = MistralTransformationModel(
            embedding_dim=proj_dimension,
            llm_hidden_size=mistral_model.config.text_config.hidden_size,
            num_prompt_tokens=K_TOKENS,
            num_mapped_tokens=K_TOKENS
        ).to(device).eval()
        transform_model.load_state_dict(torch.load(decoder_path, map_location=device))
        transform_models[embedding_type] = transform_model

    print("Preparing embeddings and summaries for shuffling...")
    embeddings_by_type = {etype: [] for etype in embedding_types}
    summaries_by_type = {etype: [] for etype in embedding_types}

    with torch.no_grad():
        for sample_id, sample_data in tqdm(eval_data.items(), desc="Extracting embeddings and summaries"):
            abstract = sample_data["abstract"].strip()
            inputs = unified_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512).to(device)
            unified_embeddings = unified_model(inputs['input_ids'], inputs['attention_mask'])

            for etype in embedding_types:
                summaries = sample_data.get(etype, [])

                valid_summaries = [s for s in summaries if s and "not applicable" not in s.lower()]
                if not valid_summaries:
                    continue

                embedding = unified_embeddings[etype.split("_")[0]]

                embeddings_by_type[etype].append(embedding)
                summaries_by_type[etype].append(valid_summaries)

    print("Shuffling embedding-summary pairs...")
    results = {etype: {'losses': [], 'perplexities': []} for etype in embedding_types}

    for etype in embedding_types:
        if not embeddings_by_type[etype]:
            continue

        shuffled_indices = list(range(len(summaries_by_type[etype])))
        random.shuffle(shuffled_indices)
        shuffled_summary_lists = [summaries_by_type[etype][i] for i in shuffled_indices]

        print(f"Evaluating shuffled {etype} embeddings...")
        with torch.no_grad():
            for embedding, shuffled_summary_list in tqdm(zip(embeddings_by_type[etype], shuffled_summary_lists), desc=f"Evaluating aspect {etype}"):

                sample_losses = []
                sample_perplexities = []
                for summary in shuffled_summary_list:
                    loss = calculate_mistral_reconstruction_loss(
                        mistral_model, mistral_processor, transform_models[etype],
                        embedding, [summary], device
                    )
                    if not torch.isnan(loss):
                        loss_val = loss.item()
                        perplexity = torch.exp(loss).item()
                        sample_losses.append(loss_val)
                        sample_perplexities.append(perplexity)

                if sample_losses:
                    results[etype]['losses'].append(np.mean(sample_losses))
                    results[etype]['perplexities'].append(np.mean(sample_perplexities))

    print("\n--- Mistral Decoder Shuffled Results ---")
    for etype, metrics in results.items():
        if metrics['losses']:
            avg_loss = np.mean(metrics['losses'])
            avg_perplexity = np.mean(metrics['perplexities'])
            print(
                f"  - {etype}: Average Loss = {avg_loss:.4f}, Average Perplexity = {avg_perplexity:.4f} (over {len(metrics['losses'])} samples)")
    print("-" * 47)


# Evaluates embeddings after projection and reconstruction into t-SNE visualization
def evaluate_tsne_roundtrip(model_type="llama", limit_per_aspect=None):
    assert model_type in ("llama", "mistral"), "model_type must be 'llama' or 'mistral'"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== t-SNE roundtrip evaluation ({model_type}) ===")

    if model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(LLAMA_CHECKPOINT, torch_dtype=torch.bfloat16,
                                                   device_map="auto").eval()
    else:
        # mistral
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        llm = AutoModelForImageTextToText.from_pretrained(
            MISTRAL_CHECKPOINT, quantization_config=quant_config, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True).eval()
        tokenizer = AutoProcessor.from_pretrained(MISTRAL_CHECKPOINT, trust_remote_code=True)
        if tokenizer.tokenizer.pad_token is None:
            tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    unified_model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to(device).eval()
    unified_model.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=device))
    unified_tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    eval_data = load_evaluation_data()
    embedding_types = list(prompts.keys())

    transform_models = {}
    for embedding_type in embedding_types:
        if model_type == "llama":
            decoder_path = os.path.join(LLAMA_DECODER_DIR, f"llama_decoder_{embedding_type}.pth")
            tm = LlamaTransformationModel(
                embedding_dim=proj_dimension,
                llama_hidden_size=llm.config.hidden_size,
                model_dtype=llm.dtype,
                decoded_tokens=K_TOKENS
            ).to(device).eval()
        else:
            decoder_path = os.path.join(MISTRAL_DECODER_DIR, f"decoder_{embedding_type}.pth")
            tm = MistralTransformationModel(
                embedding_dim=proj_dimension,
                llm_hidden_size=llm.config.text_config.hidden_size,
                num_prompt_tokens=K_TOKENS,
                num_mapped_tokens=K_TOKENS
            ).to(device).eval()
        tm.load_state_dict(torch.load(decoder_path, map_location=device))
        transform_models[embedding_type] = tm

    samples_by_type = _collect_samples_by_type(unified_model, unified_tokenizer, eval_data, embedding_types, device)

    if model_type == "llama":
        def compute_loss_single(tm, emb, summary):
            return calculate_llama_reconstruction_loss(llm, tokenizer, tm, emb, [summary])
    else:
        def compute_loss_single(tm, emb, summary):
            return calculate_mistral_reconstruction_loss(llm, tokenizer, tm, emb, [summary], device)

    for etype in embedding_types:
        sample_list = samples_by_type.get(etype, [])
        n = len(sample_list)
        if n == 0:
            print(f"Skipping {etype}: no valid samples.")
            continue

        print(f"\n--- Roundtrip for aspect '{etype}' (n={n}) ---")
        indices = list(range(n))
        if limit_per_aspect is not None:
            indices = indices[:limit_per_aspect]

        direct_losses, direct_perps = [], []
        round_losses, round_perps = [], []

        all_embeddings = [s[1] for s in sample_list]

        for idx in tqdm(indices, desc=f"{etype} LOO"):
            sample_id, emb_tensor, summaries = sample_list[idx]

            # build (n-1) embeddings
            other_embeddings = [all_embeddings[i] for i in range(n) if i != idx]
            embeddings_except = torch.cat(other_embeddings, dim=0).to(device)  # (n-1, D)

            # compute t-SNE on other samples:
            t_sne_coords, _ = calculate_t_SNE_embeddings(embeddings_except)

            # embed left-out sample in t-SNE
            embedded_2d = embed_new_point(embeddings_except, t_sne_coords, emb_tensor)
            embedded_2d = np.asarray(embedded_2d).reshape(1, -1)

            # map 2D -> recovered high-dim
            try:
                recovered_hdim = optimize_embedding_for_t_SNE(embeddings_except, t_sne_coords, embedded_2d.squeeze())
                if isinstance(recovered_hdim, np.ndarray):
                    recovered_hdim = torch.tensor(recovered_hdim, dtype=torch.float32, device=device)
                else:
                    recovered_hdim = recovered_hdim.to(device)
                if recovered_hdim.ndim == 1:
                    recovered_hdim = recovered_hdim.unsqueeze(0)
            except Exception as e:
                print(f"optimize_embedding_for_t_SNE failed for {etype}, sample {sample_id}: {e}")
                continue

            # decode original and recovered embeddings and accumulate metrics (average per-sample across summaries)
            tm = transform_models[etype]
            sample_direct_losses, sample_direct_perps = [], []
            sample_round_losses, sample_round_perps = [], []

            with torch.no_grad():
                for summary in summaries:
                    try:
                        loss_direct = compute_loss_single(tm, emb_tensor.to(device), summary)
                        loss_round = compute_loss_single(tm, recovered_hdim.to(device), summary)
                    except Exception as e:
                        print(f"Decoding error for sample {sample_id}, aspect {etype}: {e}")
                        continue

                    if torch.isnan(loss_direct) or torch.isnan(loss_round):
                        continue

                    sample_direct_losses.append(loss_direct.item())
                    sample_direct_perps.append(float(torch.exp(loss_direct).item()))
                    sample_round_losses.append(loss_round.item())
                    sample_round_perps.append(float(torch.exp(loss_round).item()))

            if sample_direct_losses:
                direct_losses.append(np.mean(sample_direct_losses))
                direct_perps.append(np.mean(sample_direct_perps))
            if sample_round_losses:
                round_losses.append(np.mean(sample_round_losses))
                round_perps.append(np.mean(sample_round_perps))

        def _mean_or_nan(lst):
            return float(np.mean(lst)) if lst else float("nan")

        print(f"Aspect '{etype}': evaluated samples (direct={len(direct_losses)}, roundtrip={len(round_losses)})")
        if direct_losses:
            print(
                f"  DIRECT  - Avg Loss = {_mean_or_nan(direct_losses):.4f}, Avg Perplexity = {_mean_or_nan(direct_perps):.4f}")
        if round_losses:
            print(
                f"  ROUNDTRIP - Avg Loss = {_mean_or_nan(round_losses):.4f}, Avg Perplexity = {_mean_or_nan(round_perps):.4f}")

    print("\n--- t-SNE roundtrip evaluation finished ---")


# This is the "unconditioned" ablation
def evaluate_models_unconditioned():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n--- Evaluating models without embeddings (Llama + Mistral) ---")

    print("Loading Llama model (for direct perplexities)...")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_CHECKPOINT, torch_dtype=torch.bfloat16,
                                                       device_map="auto").eval()

    print("Loading Mistral model (for chat perplexities)...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    mistral_model = AutoModelForImageTextToText.from_pretrained(
        MISTRAL_CHECKPOINT, quantization_config=quant_config, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    ).eval()
    mistral_processor = AutoProcessor.from_pretrained(MISTRAL_CHECKPOINT, trust_remote_code=True)
    if mistral_processor.tokenizer.pad_token is None:
        mistral_processor.tokenizer.pad_token = mistral_processor.tokenizer.eos_token

    eval_data = load_evaluation_data()
    embedding_types = list(prompts.keys())
    results = {etype: {'llama_losses': [], 'llama_perps': [], 'mistral_losses': [], 'mistral_perps': []} for etype in
               embedding_types}

    with torch.no_grad():
        for sample_id, sample_data in tqdm(eval_data.items(), desc="Evaluating samples (no-embedding)"):

            for etype in embedding_types:
                summaries = sample_data.get(etype, [])

                valid_summaries = [s for s in summaries if s and "not applicable" not in s.lower()]
                if not valid_summaries:
                    continue

                for summary in valid_summaries:
                    try:
                        enc = llama_tokenizer(summary, truncation=True, max_length=512, return_tensors="pt",
                                              padding=True).to(device)
                        input_ids = enc["input_ids"]
                        attention_mask = enc.get("attention_mask", None)
                        labels = torch.clone(input_ids)
                        labels[0, 0] = -100

                        outputs = llama_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        if not torch.isnan(loss):
                            loss_val = loss.item()
                            perp = float(torch.exp(loss).item())
                            results[etype]['llama_losses'].append(loss_val)
                            results[etype]['llama_perps'].append(perp)
                    except Exception as e:
                        print(f"Llama direct loss failed for sample {sample_id}, aspect {etype}: {e}")
                        continue

                for summary in valid_summaries:
                    try:
                        assistant_text = summary[:1200]
                        conversation = [
                            {"role": "system", "content": ""},
                            {"role": "user", "content": ""},
                            {"role": "assistant", "content": assistant_text}
                        ]
                        full_text = mistral_processor.apply_chat_template(conversation, tokenize=False,
                                                                          add_generation_prompt=False)
                        full_encoding = mistral_processor(text=[full_text], truncation=True, max_length=512,
                                                          return_tensors="pt", padding=True, add_special_tokens=False,
                                                          padding_side="right").to(device)
                        full_input_ids = full_encoding["input_ids"]
                        attention_mask = full_encoding["attention_mask"]
                        labels = torch.clone(full_input_ids)
                        labels[0, :5] = -100
                        labels[0, -1] = -100

                        out = mistral_model(input_ids=full_input_ids, attention_mask=attention_mask, labels=labels)
                        loss = out.loss
                        if not torch.isnan(loss):
                            loss_val = loss.item()
                            perp = float(torch.exp(loss).item())
                            results[etype]['mistral_losses'].append(loss_val)
                            results[etype]['mistral_perps'].append(perp)
                    except Exception as e:
                        print(f"Mistral chat loss failed for sample {sample_id}, aspect {etype}: {e}")
                        continue

    print("\n--- No-Embedding Results ---")
    for etype, metrics in results.items():
        printed = False
        if metrics['llama_losses']:
            printed = True
            avg_loss = np.mean(metrics['llama_losses'])
            avg_perp = np.mean(metrics['llama_perps'])
            print(
                f"  - {etype} (Llama): Avg Loss = {avg_loss:.4f}, Avg Perplexity = {avg_perp:.4f} (over {len(metrics['llama_losses'])} samples)")
        if metrics['mistral_losses']:
            printed = True
            avg_loss = np.mean(metrics['mistral_losses'])
            avg_perp = np.mean(metrics['mistral_perps'])
            print(
                f"  - {etype} (Mistral): Avg Loss = {avg_loss:.4f}, Avg Perplexity = {avg_perp:.4f} (over {len(metrics['mistral_losses'])} samples)")
        if not printed:
            print(f"  - {etype}: no valid samples processed.")
    print("-" * 40)