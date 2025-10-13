import os
from medical_embeddings.auxiliary import relative_path
import sys
import json
import time
from tqdm import tqdm
from llama_cpp import Llama

system_prompt = (
    """You are a precise summary generator for scientific abstracts in the medical domain.
You will receive a medical/scientific abstract and a single requested aspect to summarise.
For that single aspect produce one short, self-contained, declarative sentence that describes the aspect as it is addressed in the study.
Important constraints (must follow exactly):
- Do NOT output the literal common name of the aspect when that would be just naming it (e.g., a disease name or medication); instead produce a descriptive, semantic phrase that captures the concept without using the exact name or a trade/proprietary name.)
- Do NOT reference "the paper", "the authors", or phrases like "the study", "we", or "this paper".
- Your summary must only contain information about that aspect and must not include details that belong to other aspects.
If the requested aspect does not apply, return exactly: "Not applicable." wihting the json file.
Return a strict JSON object exactly in the format specified below."""
)

prompts = {

    "disease": (
            "the disease",
            (
                """Produce a single short, self-contained, declarative sentence that describes the medical condition(s) or clinical syndrome that is the focus of the abstract.
Focus on disease phenotype and pathophysiology at a general clinical level (disease class, dominant pathological process, or hallmark clinical features), and do not include information about the intervention, study design, patient selection, or measured outcomes.
Do NOT use the disease's common name, ICD code, or eponym verbatim; instead use a semantic descriptive phrase that captures the condition.
Avoid numeric values, place names, drug/device names, and references to the authors or paper.
If not applicable, return exactly: "Not applicable.".
Examples:
- \"A chronic inflammatory airway disorder characterized by progressive airflow limitation and exertional breathlessness.\"
- \"An acute ischemic cerebrovascular syndrome caused by sudden reduction in cerebral perfusion leading to focal neurological deficits.\"
- \"Neurodegenerative disorder characterized by progressive cognitive decline and memory impairment.\""""
            )
        ),

    "methodology": (
            "the methodology",
            (
                """Provide a single concise sentence describing the broad study design, data sources, and general analysis framework used in the work.
Mention design class (e.g., randomized controlled trial, prospective cohort, retrospective case-control, cross-sectional survey, systematic review and meta-analysis, in vitro assay, animal model, computational modeling), the type of data collected (e.g., clinical measures, imaging, laboratory biomarkers, electronic health records), and the overall analytic approach (e.g., longitudinal follow-up, comparative analysis, predictive modeling), but DO NOT mention the specific intervention, drug, or device, nor details about the patient population or the measured outcomes.
Avoid precise timeframes, exact sample sizes, trial registry IDs, or numerical endpoints.
If not applicable, return exactly: "Not applicable.".
Examples:
- \"A multicenter, randomized, double-blind, placebo-controlled clinical trial with parallel arms and predefined clinical endpoints.\"
- \"A prospective cohort study with standardized clinical assessments and longitudinal follow-up using registry and electronic health record data for comparative analysis.\"
- \"Systematic review and meta-analysis synthesizing results from multiple randomized controlled trials using standardized outcome measures.\""""
            )
        ),

    "patient_group": (
            "the patient group",
            (
                """Provide a single concise sentence that characterizes the population under study (human or animal) using demographic and clinical descriptors: age group, sex when relevant, general health status or risk profile, severity or stage descriptors, comorbidity burden, or whether the subjects are experimental animals.
Crucially, do NOT mention the disease/condition under study, the intervention, or the study design or outcomes.
Avoid precise ages, exact sample sizes, geographic identifiers, or other potentially identifying information.
If not applicable, return exactly: "Not applicable.".
Examples:
- \"A large group of community-dwelling older adults with multimorbidity and indicators of frailty and polypharmacy.\"
- \"A small sample of adults of reproductive age with elevated cardiometabolic risk profiles and overweight status.\"
- \"Laboratory mice, adult male, wild-type, standard housing conditions for preclinical assessment.\"
- \"High-risk patients with advanced disease progression and compromised physiological reserve.\""""
            )
        )
}

post_prompts = [
    "Create a single comprehensive sentence that closely adheres to the requirements.",
    "Note that the given task might not apply to this study. In this case, simply reply \"Not applicable.\".\n",
    "Return just the single sentence and nothing else. Create a self-contained sentence that makes sense without any additional context and summarizes the relevant factors without referencing \"the study\" or \"the paper\" (as done by the examples).\nUse the following json response format: {\"sentence\": \"Your sentence.\"}"
]

output_path = relative_path(f"../data/mistral_medical_summaries.json")

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
        n_ctx=6000,
        verbose=False
    )

    return llm, model_params


def generate_sentence_for_prompt(llm, model_params, abstract_text, prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"This is the relevant medical abstract:\n{abstract_text}\n\n"
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
        return num_prompts * 4

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
    sample_entry["abstract"] = abstract_text

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

def load_medical_abstracts(json_file="pubmed_15500_abstracts.json"):
    with open(relative_path(os.path.join("../data", json_file)), 'r', encoding='utf-8') as f:
        data = json.load(f)

    abstracts_dict = {}
    for item in data:
        abstracts_dict[item['index']] = {
            'index': item['index'],
            'formatted_text': item['formatted_text']
        }
    return abstracts_dict

def generate_sentences_for_dataset():
    medical_abstracts = load_medical_abstracts()

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    print("Loading Mistral model...")
    llm, model_params = load_mistral_model()
    print("Model loaded successfully!")

    total_samples = len(medical_abstracts)
    total_missing_work = 0

    for sample in medical_abstracts.values():
        missing = count_missing_work(existing_data, sample['index'], len(prompts))
        total_missing_work += missing

    print(f"Total missing sentences to generate: {total_missing_work}")

    work_generated = 0
    work_remaining = total_missing_work
    previous_work = work_generated
    start_time = time.time()

    pbar = tqdm(total=total_samples, desc=f"Generating sentences", file=sys.stdout)

    for sample in medical_abstracts.values():
        sample_index = sample['index']
        abstract_text = sample.get("formatted_text", "")

        if not abstract_text:
            pbar.update(1)
            continue

        missing_work = count_missing_work(existing_data, sample_index, len(prompts))

        if missing_work == 0:
            pbar.update(1)
            continue

        completed_work = process_sample(llm, model_params, existing_data, sample_index, abstract_text)
        work_generated += completed_work
        work_remaining -= completed_work

        if work_generated > previous_work + 300 or True:
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