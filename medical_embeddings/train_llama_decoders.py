import os
import random
import torch
from tqdm import tqdm
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from medical_embeddings.models import UnifiedEmbeddingModel, LlamaTransformationModel
from medical_embeddings.auxiliary import relative_path, load_medical_dataset
from settings import proj_dimension, encoder_checkpoint
from medical_embeddings.generate_aspect_specific_summaries import prompts

prompt_identifiers = list(prompts.keys())
LLAMA_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
UNIFIED_MODEL_SAVE_PATH = relative_path(f"../data/saved_models/medical/unified_embedding_model.pkl")
LLAMA_DECODER_SAVE_DIR = relative_path("../data/saved_models/medical/llama_decoders")
UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH = relative_path(f"../data/precomputed_unified_embeddings_medical.pkl")

BATCH_SIZE = 4
PATIENCE = 5
K_TOKENS = 5

LR = 1e-3
WEIGHT_DECAY = 1e-4

def create_unified_precomputed_embeddings(original_dataset, device):
    if os.path.exists(UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH):
        print(f"Loading existing unified precomputed embeddings from {UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH}...")
        with open(UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)

    print("Creating new precomputed embeddings using the Unified Model...")

    unified_model = UnifiedEmbeddingModel(encoder_checkpoint, prompt_identifiers, proj_dimension).to(device)
    unified_model.load_state_dict(torch.load(UNIFIED_MODEL_SAVE_PATH, map_location=device))
    unified_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)

    precomputed_data = {}
    with torch.no_grad():
        for sample_id, sample_data in tqdm(original_dataset.items(), desc="Generating Unified Embeddings"):
            abstract = sample_data["abstract"]
            inputs = tokenizer(abstract, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

            unified_embeddings = unified_model(inputs['input_ids'], inputs['attention_mask'])

            sample_output = {"embeddings": {}, "summaries": {}}
            for key, embedding_tensor in unified_embeddings.items():
                sample_output["embeddings"][key] = embedding_tensor.squeeze(0).cpu().numpy()

                summaries = [s for s in sample_data.get(key, []) if "not applicable" not in s.lower()]
                if summaries:
                    sample_output["summaries"][key] = summaries

            for key in sample_data:
                if key not in sample_output["embeddings"] and key not in ["abstract"]:
                    summaries = [s for s in sample_data.get(key, []) if "not applicable" not in s.lower()]
                    if summaries:
                        sample_output["summaries"][key] = summaries

            precomputed_data[sample_id] = sample_output

    print(f"Saving unified precomputed embeddings to {UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH}...")
    with open(UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(precomputed_data, f)

    return precomputed_data


def load_llama_model():
    print(f"Loading Llama model from {LLAMA_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_CHECKPOINT,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    generation_config = GenerationConfig(
        max_new_tokens=100,
        min_new_tokens=10,
        num_beams=1,
        no_repeat_ngram_size=5,
        early_stopping=True,
        output_scores=False,
        length_penalty=1.0,
        repetition_penalty=1.0,
        top_k=3,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    return model, tokenizer, generation_config


def get_training_data(embedding_type, unified_precomputed_data):
    data = []
    for sample_id, precomputed_sample in unified_precomputed_data.items():
        if (embedding_type in precomputed_sample["embeddings"] and embedding_type in precomputed_sample["summaries"]):
            embedding = precomputed_sample["embeddings"][embedding_type]
            target_sentences = precomputed_sample["summaries"][embedding_type]

            if target_sentences:
                data.append((torch.tensor(embedding, dtype=torch.float32), target_sentences))

    random.shuffle(data)
    train_data = data[150:]
    val_data = data[:150]

    print(f"Prepared data for '{embedding_type}': {len(train_data)} training samples, {len(val_data)} validation samples.")
    return train_data, val_data


def calculate_llama_reconstruction_loss(llama_model, tokenizer, transform_model, input_embeddings, target_texts):
    transformed_embeddings = transform_model(input_embeddings)

    target_encoding = tokenizer(
        target_texts,
        truncation=True,
        max_length=256,
        return_tensors="pt",
        padding=True).to(transformed_embeddings.device)

    target_encoding_plus_eos = torch.concatenate([target_encoding["input_ids"], torch.tensor(tokenizer.eos_token_id, device=llama_model.device).reshape(1, 1).repeat(target_encoding["input_ids"].shape[0], 1)], dim=-1)
    target_embeddings = llama_model.get_input_embeddings()(target_encoding_plus_eos)

    concat_embeddings = torch.cat([
        target_embeddings[:, 0].unsqueeze(1),
        transformed_embeddings,
        target_embeddings[:, 1:],
    ], dim=1)

    labels = torch.cat([
        torch.full((len(target_texts), transformed_embeddings.shape[1] + 1),
                   -100, dtype=torch.long, device=transformed_embeddings.device),
        target_encoding_plus_eos[:, 1:]
    ], dim=1)

    for x in range(labels.shape[0]):
        idx = labels.shape[1]-1
        while labels[x][idx-1] == tokenizer.pad_token_id or labels[x][idx-1] == tokenizer.eos_token_id:
            labels[x][idx] = -100
            idx -= 1

    outputs = llama_model(inputs_embeds=concat_embeddings, labels=labels)
    return outputs.loss


def generate_text_from_embedding(llama_model, tokenizer, transform_model, input_embedding, generation_config):
    llama_model.eval()
    transform_model.eval()

    with torch.no_grad():
        transformed_embedding = transform_model(input_embedding.unsqueeze(0))
        bos_embedding = llama_model.get_input_embeddings()(torch.tensor(tokenizer.bos_token_id, device=llama_model.device).reshape(1, -1))
        concat_embeddings = torch.cat([
            bos_embedding,
            transformed_embedding], dim=1)
        output_ids = llama_model.generate(
            inputs_embeds=concat_embeddings,
            generation_config=generation_config
        )

        generated_texts = [tokenizer.decode(output_ids[k], skip_special_tokens=True) for k in range(min(len(output_ids), 3))]

    return generated_texts


def evaluate_llama_decoder(llama_model, tokenizer, transform_model, val_data, device):
    llama_model.eval()
    transform_model.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(val_data), BATCH_SIZE), desc="Evaluating", leave=False):
            batch = val_data[i:i + BATCH_SIZE]
            batch = [(x, y[0]) for x, y in batch]
            if not batch:
                continue

            embeddings, texts = zip(*batch)
            embedding_batch = torch.stack(embeddings).to(device)

            loss = calculate_llama_reconstruction_loss(llama_model, tokenizer, transform_model, embedding_batch, list(texts))
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def train_llama_decoder_for_type(embedding_type, llama_model, tokenizer, generation_config, train_data, val_data):
    print(f"\n--- Starting Llama training for embedding type: {embedding_type} ---")

    device = next(llama_model.parameters()).device
    llama_hidden_size = llama_model.config.hidden_size

    transform_model = LlamaTransformationModel(
        embedding_dim=proj_dimension,
        model_dtype=llama_model.dtype,
        llama_hidden_size=llama_hidden_size,
        decoded_tokens=K_TOKENS
    ).to(device)

    optimizer = torch.optim.AdamW(transform_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    llama_model.eval()

    best_val_loss = evaluate_llama_decoder(llama_model, tokenizer, transform_model, val_data, device)
    epochs_no_improve = 0

    batches_per_epoch = 250
    total_epochs = 100

    print(f"Training for {total_epochs} epochs with {batches_per_epoch} batches per epoch")

    def infinite_data_iterator(data):
        while True:
            random.shuffle(data)
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE]
                batch = [(x, random.choice(y)) for x, y in batch]
                if batch:
                    yield batch

    data_iter = infinite_data_iterator(train_data)

    for epoch in range(total_epochs):
        transform_model.train()
        epoch_loss = 0

        for batch_num in tqdm(range(batches_per_epoch), desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False):
            batch = next(data_iter)
            embeddings, texts = zip(*batch)
            embedding_batch = torch.stack(embeddings).to(device)

            optimizer.zero_grad()

            loss = calculate_llama_reconstruction_loss(llama_model, tokenizer, transform_model, embedding_batch, list(texts))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / batches_per_epoch
        print(f"Epoch {epoch + 1}: Training Loss = {avg_epoch_loss:.4f}")

        val_loss = evaluate_llama_decoder(llama_model, tokenizer, transform_model, val_data, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(LLAMA_DECODER_SAVE_DIR, f"llama_decoder_{embedding_type}.pth")
            torch.save(transform_model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Finished training for {embedding_type}. Best validation loss: {best_val_loss:.4f}")
    return best_val_loss


def train_all_llama_decoders():
    os.makedirs(LLAMA_DECODER_SAVE_DIR, exist_ok=True)

    llama_model, tokenizer, generation_config = load_llama_model()
    device = next(llama_model.parameters()).device

    print("Loading datasets...")
    dataset = load_medical_dataset()
    dataset = {key: dataset[key] for key in list(dataset.keys())[500:]}

    unified_precomputed_data = create_unified_precomputed_embeddings(dataset, device)

    for embedding_type in prompt_identifiers:
        save_path = os.path.join(LLAMA_DECODER_SAVE_DIR, f"llama_decoder_{embedding_type}.pth")
        if os.path.exists(save_path):
            print(f"Decoder for '{embedding_type}' already exists. Skipping.")
            continue

        train_data, val_data = get_training_data(embedding_type, unified_precomputed_data)

        train_llama_decoder_for_type(embedding_type, llama_model, tokenizer, generation_config, train_data, val_data)

def test_llama_decoder_generation(embedding_type, num_samples=5):
    print(f"\n--- Testing Llama decoder for '{embedding_type}' ---")

    decoder_path = os.path.join(LLAMA_DECODER_SAVE_DIR, f"llama_decoder_{embedding_type}.pth")
    if not os.path.exists(decoder_path):
        print(f"No trained decoder found for '{embedding_type}'")
        return

    llama_model, tokenizer, generation_config = load_llama_model()
    device = next(llama_model.parameters()).device

    llama_hidden_size = llama_model.config.hidden_size

    transform_model = LlamaTransformationModel(
        embedding_dim=proj_dimension,
        model_dtype=llama_model.dtype,
        llama_hidden_size=llama_hidden_size,
        decoded_tokens=K_TOKENS
    ).to(device)

    transform_model.load_state_dict(torch.load(decoder_path, map_location=device))

    dataset = load_medical_dataset()
    dataset = {key: dataset[key] for key in list(dataset.keys())[500:]}

    unified_precomputed_data = create_unified_precomputed_embeddings(dataset, device)

    test_samples = list(unified_precomputed_data.items())[:num_samples]

    for sample_id, sample_data in test_samples:
        if embedding_type not in sample_data["embeddings"]:
            continue

        print(f"\nSample ID: {sample_id}")

        embedding = torch.tensor(sample_data["embeddings"][embedding_type], dtype=torch.float32).to(device)
        generated_texts = generate_text_from_embedding(llama_model, tokenizer, transform_model, embedding, generation_config)

        print("Generated texts:")
        for i, text in enumerate(generated_texts, 1):
            print(f"  {i}: {text.strip()}")

        if embedding_type in sample_data["summaries"]:
            ground_truth = sample_data["summaries"][embedding_type][0]
            print(f"Ground truth: {ground_truth}")

            try:
                with torch.no_grad():
                    ground_truth_loss = calculate_llama_reconstruction_loss(
                        llama_model, tokenizer, transform_model,
                        embedding.unsqueeze(0), [ground_truth]
                    )
                    print(f"Ground truth reconstruction loss: {ground_truth_loss.item():.4f}")
            except Exception as e:
                pass

        print("-" * 50)

def test_all_llama_decoders(num_samples=3):
    embedding_types_to_test = prompt_identifiers

    for embedding_type in embedding_types_to_test:
        decoder_path = os.path.join(LLAMA_DECODER_SAVE_DIR, f"llama_decoder_{embedding_type}.pth")
        if os.path.exists(decoder_path):
            test_llama_decoder_generation(embedding_type, num_samples)
        else:
            print(f"No trained decoder found for '{embedding_type}'. Skipping test.")