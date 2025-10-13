import os
import sys
import json
import time
from tqdm import tqdm
from llama_cpp import Llama
from invasion_biology_embeddings.auxiliary import relative_path, load_invasion_dataset

system_prompt = (
    "You are a precise summary generator for scientific abstracts from the field of invasion biology. "
    "You will receive a scientific abstract from the field of invasion biology. For a single requested aspect, produce short, self-contained, declarative sentences that describe "
    "the aspect (as it is addressed in the scientific study) in general terms (no precise species names, no very specific place names, no numeric values, no references to 'the study', "
    "'we', 'this paper', or authors). If the requested aspect does not apply, return exactly: \"Not applicable.\". "
    "Return a strict JSON object as requested in the user instruction (no extra commentary)."
)

prompts = {
    "hypothesis_general": (
        "the relationship (general)",
        (
            "Produce a single short declarative sentence that captures the broad, commonly-studied ecological "
            "relationship or directional hypothesis from the field of invasion biology that is addressed by the abstract. Use general causal or correlational "
            "language (e.g., 'increases', 'reduces', 'facilitates', 'is associated with') if applicable and avoid specific species names (general classes are okay), "
            "place names, numeric values, and any phrasing that references the paper or authors. "
            "Example: \"Greater propagule pressure increases establishment probability across habitat types.\""
        )
    ),

    "hypothesis_specific": (
        "the relationship (specific)",
        (
            "Produce a single short declarative sentence that captures the general relationship or hypothesis commonly addressed in the field of invasion biology "
            "that is addressed in this scientific study. Try to abstract away from the specific details of the study to create a more general statement about the relationship. Example: "
            "\"Plants may experience increased survival and growth in introduced ranges due to reduced impacts from local herbivores and pathogens compared to their native ranges.\""
        )
    ),

    "ecosystem": (
        "the ecosystem",
        (
            "Provide a concise but informative sentence describing the ecosystem type(s) studied using broad ecological "
            "categories and invasion-relevant context. Include all factors that are relevant in the context of invasion biology, "
            "which might include climate regime (e.g., temperate, tropical, arid), dominant "
            "habitat structure (e.g., freshwater stream, coastal saltmarsh, temperate grassland, urban green space), or "
            "ecological features that affect invasibility (e.g., high native species richness, frequent disturbance, "
            "hydrological variability, habitat fragmentation, salinity gradient, general classes of species that exist in this habitat). "
            "Avoid specific place names, specific species names, and numeric values. "
            "If a very broad region without a precise set of common characteristics is studies, then name this region in a descriptive sentence."
            " Examples:\n"
            "- \"Temperate grassland ecosystems with scattered, open woodland and a shallow soil layer over limestone bedrock.\"\n"
            "- \"Coastal saltmarsh in a temperate climate with strong salinity gradients and frequent anthropogenic disturbance.\"\n"
            "- \"Large areas of grasslands and forests across eastern Europe.\""
        )
    ),

    "researchquestion": (
        "research question",
        (
            "State the central, most generalizable research question or comparative focus of the scientific study as a single declarative sentence. "
            "Longer sentences are acceptable — emphasize the experimental or observational contrast, the response measured, "
            "and the general context (e.g., spatial or temporal scales, or interacting factors) without naming specific species or specific places. "
            "Examples:\n"
            "- \"Does disturbance frequency and landscape connectivity together determine establishment and spread of "
            "non-native plants across heterogeneous temperate grasslands?\"\n"
            "- \"How does the abundance of a non-native predator influence recruitment and survival of native prey across habitat "
            "gradients over multiple reproductive seasons?\""
        )
    ),

    "species": (
        "the species",
        (
            "Describe the focal organism(s) that are central to this study in functional, non-identifying terms. "
            "The description might contain state taxonomic level (plant, insect, fish, bird), "
            "growth or body form, trophic role (herbivore, omnivore, predator, detritivore), broad climate/biome affinity (temperate, "
            "tropical, boreal, arid), dispersal or reproductive mode (wind-dispersed seeds, planktonic larvae, broadcast spawner, "
            "clonal spread), and traits relevant to species invasions (e.g., generalist diet, high reproductive output, tolerance to salinity). "
            "Avoid specific species names, place names, and numeric measures. Examples:\n"
            "- \"Perennial herbaceous plant from temperate regions with clonal spread and wind-dispersed seeds.\"\n"
            "- \"Small-bodied benthic fish in temperate estuaries, generalist predator on invertebrates with planktonic larvae.\"\n"
            "Focus solely on the species that is central to this study. If multiple species are important, include all in a single comprehensive sentence."
        )
    ),

    "methodology": (
    "the methodology",
    (
        "Summarize the methodological approach underlying the study in a single sentence that describes: "
        "the general study design (observational, manipulative/experimental, modeling), "
        "the type of data collected (field surveys, manipulative plot experiment, lab trials, remote sensing, genetic data), "
        "and the overall analysis style or framework (comparative, longitudinal, predictive modeling, etc.). "
        "Focus strictly on the raw structure of the methodology without mentioning specific species, ecosystems, variables, measured outcomes, or desired results. "
        "Sentences should be complete and moderately detailed (more than just a few words), but should remain fully generic. "
        "Examples:\n"
        "- \"A manipulative field experiment applying controlled treatments across multiple replicated plots to assess responses over several growing seasons.\"\n"
        "- \"An observational study combining standardized transect surveys with long-term environmental monitoring records for comparative temporal analysis.\"\n"
        "- \"A meta-analytical synthesis integrating results from multiple independent field experiments using standardized statistical frameworks.\""
    )
),


    "recommendation": (
        "the recommendations",
        (
            "If the abstract states explicit management or policy recommendations, return a single concise sentence summarizing them "
            "without referring to the paper or authors. If no explicit recommendations are stated, return exactly \"Not applicable.\". "
            "Examples:\n"
            "- \"Prioritize early detection and rapid removal in high-connectivity corridors and restore native vegetation to reduce re-invasion.\"\n"
            "- \"Implement targeted removal of established populations combined with post-removal habitat restoration and monitoring.\""
        )
    ),
}

post_prompts = ["Create a single comprehensive sentence that closely adheres to the requirements. ",
                "Note that the given task might not apply to this study. In this case, simply reply \"Not applicable.\".\n",
                "Return just the single sentence and nothing else. Create a self-contained sentence that makes sense without any additional context and summarizes the relevant factors without "
                "referencing \"the study\" or \"the paper\" (as done by the examples).\nUse the following json response format: {\"sentence\": \"Your sentence.\"}"]

keyword_prompt = ("Your task is to create a list of relevant keywords for this scientific study. "
                  "Focus just on the most relevant keywords that distinguish this study from other studies in the field of invasion biology and that would be relevant search terms used by scientists in this field.\n"
                  "Use the following json response format: {\"keywords\": [\"Keyword 1\", \"Keyword 2\"]}")

output_path = relative_path(f"../data/mistral_invasion_biology_summaries.json")

def load_mistral_model(model_path="bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF"):
    parameters_map = {
        "Mistral": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "min_p": 0.01
        }
    }
    model_params = parameters_map["Mistral"]

    llm = Llama.from_pretrained(
        repo_id=model_path,
        filename="*Q8_0.gguf",
        n_gpu_layers=-1,
        n_ctx=4000,
        verbose=False
    )

    return llm, model_params


def generate_sentence_for_prompt(llm, model_params, abstract_text, prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"This is the relevant scientific abstract:\n{abstract_text}\n\n"
                                    f"Your task is the following:\n{prompt[1]}.\n{''.join(post_prompts)}"}
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=400,
            response_format={"type": "json_object"},
            **model_params
        )

        generated_text = response["choices"][0]["message"]["content"].strip()
        sentence = json.loads(generated_text)['sentence']
        return sentence
    except Exception as e:
        print(f"Error generating sentence: {e}")
        return None

def count_missing_work(sample_data, sample_index, num_prompts):
    if str(sample_index) not in sample_data:
        return num_prompts * 4 + 1

    sample_entry = sample_data[str(sample_index)]
    missing_count = 0

    for prompt_key in prompts:
        if prompt_key not in sample_entry:
            missing_count += 4
        else:
            current_sentences = len(sample_entry[prompt_key])
            missing_count += max(0, 4 - current_sentences)

    return missing_count

def update_progress_description(pbar, work_generated, work_remaining, start_time):
    if work_generated > 0:
        elapsed_time = time.time() - start_time
        avg_time_per_work = elapsed_time / work_generated
        estimated_remaining_time = avg_time_per_work * work_remaining

        hours = int(estimated_remaining_time // 3600)
        minutes = int((estimated_remaining_time % 3600) // 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m"

        pbar.set_description(f"Est. remaining: {time_str} ({work_remaining} items left)")
    else:
        pbar.set_description(f"Calculating estimate...")

def process_sample(llm, model_params, sample_data, sample_index, abstract_text):
    work_completed = 0

    if str(sample_index) not in sample_data:
        sample_data[str(sample_index)] = {}

    sample_entry = sample_data[str(sample_index)]
    for prompt_key, prompt in prompts.items():
        if prompt_key not in sample_entry:
            sample_entry[prompt_key] = []

        current_sentences = sample_entry[prompt_key]
        sentences_needed = 4 - len(current_sentences)

        for _ in range(sentences_needed):
            sentence = generate_sentence_for_prompt(llm, model_params, abstract_text, prompt)
            if sentence:
                sample_entry[prompt_key].append(sentence)
                work_completed += 1

    return work_completed

def get_sample_slice(samples, start_idx, end_idx):
    sample_list = list(samples.items())
    if end_idx == -1:
        end_idx = len(sample_list)
    return dict(sample_list[start_idx:end_idx])

def generate_sentences_for_dataset():
    dataset = load_invasion_dataset()

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
        existing_data["labeled"] = {}
        existing_data["unlabeled"] = {}

    print()

    print("Loading Mistral model...")
    llm, model_params = load_mistral_model()
    print("Model loaded successfully!")

    total_samples = len(dataset["labeled"]) + len(dataset["unlabeled"])
    total_missing_work = 0

    for paper_idx, sample in dataset["labeled"].items():
        missing = count_missing_work(existing_data["labeled"], sample['index'], len(prompts))
        total_missing_work += missing

    for paper_idx, sample in dataset["unlabeled"].items():
        missing = count_missing_work(existing_data["unlabeled"], sample['index'], len(prompts))
        total_missing_work += missing

    print(f"Total missing sentences to generate: {total_missing_work}")

    work_generated = 0
    work_remaining = total_missing_work
    previous_work = work_generated
    start_time = time.time()

    pbar = tqdm(total=total_samples, desc=f"Generating sentences", file=sys.stdout)

    for sample in dataset["labeled"].values():
        sample_index = sample['index']
        abstract_text = sample.get("prediction_text", "")

        if not abstract_text:
            pbar.update(1)
            continue

        missing_work = count_missing_work(existing_data["labeled"], sample_index, len(prompts))

        if missing_work == 0:
            pbar.update(1)
            continue

        completed_work = process_sample(llm, model_params, existing_data["labeled"], sample_index, abstract_text)
        work_generated += completed_work
        work_remaining -= completed_work

        if work_generated > previous_work + 300 or True:
            with open(output_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            previous_work = work_generated

        pbar.update(1)
        update_progress_description(pbar, work_generated, work_remaining, start_time)

    for paper_idx, sample in dataset["unlabeled"].items():
        sample_index = sample['index']
        abstract_text = sample.get("prediction_text", "")

        if not abstract_text:
            pbar.update(1)
            continue

        missing_work = count_missing_work(existing_data["unlabeled"], sample_index, len(prompts))

        if missing_work == 0:
            pbar.update(1)
            continue

        completed_work = process_sample(llm, model_params, existing_data["unlabeled"], sample_index, abstract_text)
        work_generated += completed_work
        work_remaining -= completed_work

        if work_generated > previous_work + 300:
            with open(output_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            previous_work = work_generated

        pbar.update(1)
        update_progress_description(pbar, work_generated, work_remaining, start_time)

    pbar.close()

    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"\nCompleted! Generated sentences saved to {output_path}")
    print(f"Total sentences generated: {work_generated}")