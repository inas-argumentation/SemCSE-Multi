import os
import random
import torch
from tqdm import tqdm
import json
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor
from invasion_biology_embeddings.models import MistralTransformationModel
from invasion_biology_embeddings.auxiliary import relative_path
from settings import proj_dimension
from invasion_biology_embeddings.train_llama_decoders import create_unified_precomputed_embeddings, get_training_data
from invasion_biology_embeddings.generate_aspect_specific_summaries import prompts

LLM_MODEL_PATH = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
UNIFIED_MODEL_SAVE_PATH = relative_path(f"../data/saved_models/invasion_biology/unified_embedding_model.pkl")
DECODER_SAVE_DIR = relative_path("../data/saved_models/invasion_biology/mistral_decoders")
DATASET_PATH = relative_path("../data/mistral_invasion_biology_summaries.json")
UNIFIED_PRECOMPUTED_EMBEDDINGS_PATH = relative_path(f"../data/precomputed_unified_embeddings_invasion_biology.pkl")

BATCH_SIZE = 4
GRAD_ACC_STEPS = 1
NUM_TRAINABLE_PROMPT_TOKENS = 5
NUM_EMBEDDING_MAPPED_TOKENS = 5
TOTAL_NUM_ADDED_TOKENS = NUM_EMBEDDING_MAPPED_TOKENS + NUM_TRAINABLE_PROMPT_TOKENS
PATIENCE = 5

prompt_identifiers = list(set([p.split("_")[0] for p in prompts.keys()]))
individual_prompt_identifiers = list(prompts.keys())

def load_llm_and_processor(model_path):
    print(f"Loading LLM from {model_path}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, processor

def calculate_reconstruction_loss(llm, processor, decoder, embedding_batch, target_texts, device):
    llm_dtype = next(llm.parameters()).dtype
    decoder_output_embeddings = decoder(embedding_batch.to(device), llm_dtype)

    batch_size = embedding_batch.size(0)
    pad_tok = processor.tokenizer.pad_token
    if pad_tok is None:
        raise ValueError("processor.tokenizer.pad_token is None")

    placeholder_user_content = ''.join([pad_tok] * TOTAL_NUM_ADDED_TOKENS)

    batch_conversations = []
    for target_text in target_texts:
        conversation = [
            {"role": "system", "content": ""},
            {"role": "user", "content": placeholder_user_content},
            {"role": "assistant", "content": target_text[:1200]}
        ]
        full_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        batch_conversations.append(full_text)

    full_encoding = processor(text=batch_conversations, truncation=True, max_length=512,
                             return_tensors="pt", padding=True, add_special_tokens=False, padding_side="right").to(device)
    full_input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]

    pad_id = processor.tokenizer.pad_token_id
    seq_to_match = torch.tensor([pad_id] * TOTAL_NUM_ADDED_TOKENS, device=device)

    placeholder_starts = []
    seq_len = full_input_ids.size(1)

    for b in range(batch_size):
        found = None
        for i in range(seq_len - TOTAL_NUM_ADDED_TOKENS + 1):
            window = full_input_ids[b, i:i + TOTAL_NUM_ADDED_TOKENS]
            if torch.equal(window, seq_to_match):
                found = i
                break
        if found is None:
            decoded = processor.tokenizer.decode(full_input_ids[b].cpu().tolist(), skip_special_tokens=False)
            raise ValueError(f"Could not find placeholder tokens for sample {b}. Example decoded: {decoded[:200]}")
        placeholder_starts.append(found)

    full_embeddings = llm.get_input_embeddings()(full_input_ids)

    inputs_embeds = full_embeddings.clone()
    if decoder_output_embeddings.dtype != inputs_embeds.dtype:
        decoder_output_embeddings = decoder_output_embeddings.to(inputs_embeds.dtype)

    for b in range(batch_size):
        start = placeholder_starts[b]
        end = start + TOTAL_NUM_ADDED_TOKENS
        inputs_embeds[b, start:end] = decoder_output_embeddings[b]

    labels = full_input_ids.clone()
    for b in range(batch_size):
        assistant_start = placeholder_starts[b] + TOTAL_NUM_ADDED_TOKENS + 1 # Plus one to exclude instruction end token
        labels[b, :assistant_start] = -100
        labels[labels.eq(processor.tokenizer.pad_token_id)] = -100

    outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def evaluate_decoder(llm, processor, decoder, val_data, device, batch_size=1):
    decoder.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(val_data), batch_size), desc="Evaluating", leave=False):
            batch = val_data[i:i + batch_size]
            batch = [(x, y[0]) for x, y in batch]

            embeddings, texts = zip(*batch)
            embedding_batch = torch.stack(embeddings)

            loss = calculate_reconstruction_loss(llm, processor, decoder, embedding_batch, list(texts), device)
            total_loss += loss.item()
            num_batches += 1

    loss = total_loss / max(1, num_batches)
    print(f"Loss: {loss:.4f}")
    return loss

def train_decoder_for_type(embedding_type, llm, processor, train_data, val_data):
    print(f"\n--- Starting training for embedding type: {embedding_type} ---")

    device = llm.device
    llm_hidden_size = llm.config.text_config.hidden_size
    embedding_dim = proj_dimension

    decoder = MistralTransformationModel(
        embedding_dim=embedding_dim,
        llm_hidden_size=llm_hidden_size,
        num_prompt_tokens=NUM_TRAINABLE_PROMPT_TOKENS,
        num_mapped_tokens=NUM_EMBEDDING_MAPPED_TOKENS).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)

    sampling_list = train_data.copy()
    random.shuffle(sampling_list)

    def get_train_batch(batch_size):
        nonlocal sampling_list

        if len(sampling_list) < batch_size:
            sampling_list = train_data.copy()
            random.shuffle(sampling_list)

        batch = sampling_list[:batch_size]
        sampling_list = sampling_list[batch_size:]
        return batch

    best_val_loss = evaluate_decoder(llm, processor, decoder, val_data, device)
    epochs_no_improve = 0

    for epoch in range(100):
        decoder.train()

        for _ in tqdm(range(250), desc=f"Epoch {epoch+1}/{100} Training Steps"):
            for _ in range(GRAD_ACC_STEPS):
                batch = get_train_batch(BATCH_SIZE)
                batch = [(x, random.choice(y)) for x, y in batch]

                embeddings, texts = zip(*batch)
                embedding_batch = torch.stack(embeddings)

                optimizer.zero_grad()

                loss = calculate_reconstruction_loss(llm, processor, decoder, embedding_batch, list(texts), device)

                loss.backward()
            optimizer.step()

        val_loss = evaluate_decoder(llm, processor, decoder, val_data, device)
        print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(DECODER_SAVE_DIR, f"decoder_{embedding_type}.pth")
            torch.save(decoder.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            break
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Finished training for {embedding_type}. Best validation loss: {best_val_loss:.4f}")

def train_all_mistral_decoders():
    os.makedirs(DECODER_SAVE_DIR, exist_ok=True)

    llm, processor = load_llm_and_processor(LLM_MODEL_PATH)
    device = llm.device

    print("Loading datasets...")
    with open(DATASET_PATH, "r") as f:
        original_dataset = json.load(f)

    unified_precomputed_data = create_unified_precomputed_embeddings(original_dataset, device)
    embedding_types_to_train = individual_prompt_identifiers

    for embedding_type in embedding_types_to_train:
        save_path = os.path.join(DECODER_SAVE_DIR, f"decoder_{embedding_type}.pth")
        if os.path.exists(save_path):
            print(f"Decoder for '{embedding_type}' already exists. Skipping.")
            continue

        train_data, val_data = get_training_data(embedding_type, unified_precomputed_data)

        if not train_data:
            print(f"No training data found for '{embedding_type}'. Skipping.")
            continue

        train_decoder_for_type(embedding_type, llm, processor, train_data, val_data)


def test_mistral_decoder_generation(embedding_type, num_samples=5):
    print(f"\n--- Testing decoder for '{embedding_type}' ---")

    decoder_path = os.path.join(DECODER_SAVE_DIR, f"decoder_{embedding_type}.pth")
    if not os.path.exists(decoder_path):
        print(f"No trained decoder found for '{embedding_type}'")
        return

    llm, processor = load_llm_and_processor(LLM_MODEL_PATH)
    device = llm.device

    llm_hidden_size = llm.config.text_config.hidden_size
    embedding_dim = proj_dimension

    decoder = MistralTransformationModel(
        embedding_dim=embedding_dim,
        llm_hidden_size=llm_hidden_size,
        num_prompt_tokens=NUM_TRAINABLE_PROMPT_TOKENS,
        num_mapped_tokens=NUM_EMBEDDING_MAPPED_TOKENS).to(device)

    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()

    with open(DATASET_PATH, "r") as f:
        original_dataset = json.load(f)
    unified_precomputed_data = create_unified_precomputed_embeddings(original_dataset, device)

    test_samples = list(unified_precomputed_data.items())[:num_samples]

    with torch.no_grad():
        for sample_id, sample_data in test_samples:
            if embedding_type.split("_")[0] not in sample_data["embeddings"]:
                continue

            print(f"\nSample ID: {sample_id}")

            embedding = torch.tensor(sample_data["embeddings"][embedding_type.split("_")[0]], dtype=torch.float32).unsqueeze(0).to(device)
            llm_dtype = next(llm.parameters()).dtype
            decoder_output = decoder(embedding, llm_dtype)

            pad_tok = processor.tokenizer.pad_token
            pad_tok_id = processor.tokenizer.pad_token_id
            if pad_tok is None:
                pad_tok = processor.tokenizer.eos_token
                pad_tok_id = processor.tokenizer.eos_token_id

            placeholder_user_content = ''.join([pad_tok] * TOTAL_NUM_ADDED_TOKENS)
            conversation = [
                {"role": "system", "content": ""},
                {"role": "user", "content": placeholder_user_content}
            ]

            prompt_text = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True)

            prompt_encoding = processor(text=prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512).to(device)
            prompt_embeds = llm.get_input_embeddings()(prompt_encoding["input_ids"])

            prompt_tokens = prompt_encoding["input_ids"][0]
            seq_to_match = torch.tensor([pad_tok_id] * TOTAL_NUM_ADDED_TOKENS, device=device)

            placeholder_start = None
            seq_len = prompt_tokens.shape[0]
            for i in range(seq_len - TOTAL_NUM_ADDED_TOKENS + 1):
                window = prompt_tokens[i:i + TOTAL_NUM_ADDED_TOKENS]
                if torch.equal(window, seq_to_match):
                    placeholder_start = i
                    break

            if placeholder_start is not None:
                inputs_embeds = prompt_embeds.clone()
                if decoder_output.dtype != inputs_embeds.dtype:
                    decoder_output = decoder_output.to(inputs_embeds.dtype)

                inputs_embeds[0, placeholder_start:placeholder_start + TOTAL_NUM_ADDED_TOKENS] = decoder_output[0]

                with torch.no_grad():
                    outputs = llm.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prompt_encoding["attention_mask"],
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.95,
                        pad_token_id=processor.tokenizer.eos_token_id)

                generated_tokens = outputs[0]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                print(f"Generated response: {generated_text.strip()}")
            else:
                print("Could not find placeholder tokens for replacement")

            if embedding_type in sample_data["summaries"]:
                print(f"Ground truth: {sample_data['summaries'][embedding_type][0]}")

            print("-" * 50)