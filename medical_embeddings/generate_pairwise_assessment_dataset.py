import os
import json
import time
import random
import sys
from itertools import combinations
from collections import OrderedDict
from tqdm import tqdm

from llama_cpp import Llama
from medical_embeddings.auxiliary import relative_path, load_medical_dataset

MODEL_REPO_ID = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
MODEL_FILENAME_GGUF = "*Q8_0.gguf"
N_CTX = 32000
TEMPERATURE = 0.8
MAX_TOKENS = 32000

SAVE_EVERY = 10
ABSTRACT_TRUNC_CHARS = 1000
OUTPUT_DIR = relative_path("../data/llm_pairwise_assessments/medical")
INPUT_SUMMARIES_JSON = relative_path("../data/mistral_medical_summaries.json")
RANDOM_SEED = 0
RETRY_MAX = 3
RETRY_BACKOFF = 1
NUMBER_OF_TEST_SAMPLES = 500
TARGET_PER_ASPECT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

ASPECTS = OrderedDict([
    ("disease", (
        "disease / medical condition",
        (
            "Compare ONLY the medical condition(s) or clinical syndrome that is the focus of each abstract. "
            "Judge similarity based on these components when explicitly present:\n"
            "(A) disease class or pathological category (e.g., cardiovascular, neurological, infectious, neoplastic, autoimmune, metabolic),\n"
            "(B) dominant pathophysiological process or mechanism (e.g., inflammation, ischemia, degeneration, infection, malignant transformation),\n"
            "(C) primary anatomical system or organ affected (e.g., respiratory, cardiovascular, nervous, musculoskeletal, specific body part, etc.),\n"
            "You may take additional factors into account of they apply to the disease or condition mentioned in the abstracts. "
            "Abstract away from specific disease names, ICD codes, eponyms, and demographic details. "
            "Focus on the underlying pathophysiology and clinical phenotype. "
            "Use only information explicitly stated in the abstracts. "
            "Do NOT up-score similarity just because both studies are medical research, since all studies you will receive are about medical research."
        ),
        [
            "6: Exact same disease.",
            "5: Highly related disease, as indicated by the same disease class AND same pathophysiological process AND similar anatomical systems.",
            "4: Same disease class with some overlap in pathophysiology OR anatomical system, but clear differences in other components.",
            "3: Same broad disease class (e.g., both cardiovascular, both neurological) but different pathophysiological processes and anatomical targets.",
            "2: Very weakly related disease, like, for example, similar pathophysiological processes, but no other similarities.",
            "1: Different disease classes or insufficient disease description to justify similarity beyond both being medical conditions."
        ]
    )),

    ("methodology", (
        "methodological structure",
        (
            "Compare ONLY the methodological approach used by each study, independent of the specific disease, intervention, patient population, or outcomes measured. "
            "Judge similarity using these factors when applicable:\n"
            "(A) Study design class (e.g., randomized controlled trial, prospective cohort, retrospective case-control, cross-sectional survey, systematic review/meta-analysis, in vitro assay, animal model, computational modeling),\n"
            "(B) Data source and collection method (e.g., clinical measures, imaging, laboratory biomarkers, electronic health records, patient-reported outcomes, tissue samples),\n"
            "(C) Analysis framework (e.g., comparative effectiveness, longitudinal follow-up, predictive modeling, causal inference, descriptive analysis),\n"
            "(D) Study scope (e.g., single-center vs multicenter, duration of follow-up, sample size category).\n\n"
            "Use only methodological details explicitly stated in the abstracts. "
            "Do not consider the specific medical condition, intervention, or patient characteristics when assessing methodological similarity."
        ),
        [
            "5: Mostly identical study design, as indicated by the same study design class AND same data collection methods AND related analysis framework and comparable scope.",
            "4: Highly related study design, as indicated by the same study design class AND same data collection methods, but different analysis frameworks and different scope.",
            "3: Related study design, as indicated by same the study design class with some overlap in data collection OR analysis approach, but clear differences in other methodological components.",
            "2: Same broad methodological family (e.g., both experimental, both observational) but different specific designs and approaches.",
            "1: Different methodological approaches or insufficient methodological detail to justify similarity beyond both being medical research."
        ]
    )),

    ("patient_group", (
        "patient population / study subjects",
        (
            "Compare ONLY the characteristics of the study population described in each abstract. "
            "Judge similarity using these factors when explicitly stated:\n"
            "(A) Age group or life stage (e.g., pediatric, adult, elderly, reproductive age),\n"
            "(B) Health status or risk profile (e.g., healthy volunteers, high-risk patients, critically ill, community-dwelling),\n"
            "(C) Disease severity, stage, or functional status when relevant (e.g., early-stage, advanced disease, treatment-naive),\n"
            "(D) Type of species (relevant for studies that focus on on animals), \n"
            "(E) rough sample size (e.g., very small study vs. large clinical trial), \n"
            "(F) Comorbidity burden or special populations (e.g., multimorbidity, immunocompromised, pregnancy).\n\n"
            "Do NOT consider the specific disease being studied, the intervention, or study outcomes. "
            "Focus only on population demographics and clinical characteristics. "
            "Abstract away from exact ages, precise sample sizes, and geographic identifiers. "
            "Use only population descriptors explicitly stated in the abstracts."
        ),
        [
            "5: Highly similar group of study subjects with most characteristics matching, for example, similarly-sized group with similar age and gender characteristics AND same health status/risk profile AND same disease severity/stage.",
            "4: Very similar group of study subjects with several factors matching.",
            "3: Somewhat similar group of study subjects, with few of these factors matching.",
            "2: Both are human studies or both are animal studies, but other than that there are no substantial similarities.",
            "1: Completely different subjects, for example, one focused on animals and one on humans."
        ]
    ))

])


def load_llm(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME_GGUF, n_ctx=N_CTX):
    params = {
        "temperature": TEMPERATURE,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0,
        "presence_penalty": 1
    }
    llm = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=False
    )
    return llm, params


def safe_load_json_from_text(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from model output.")


def load_medical_abstracts_for_evaluation(max_samples=NUMBER_OF_TEST_SAMPLES):
    dataset = load_medical_dataset()
    evaluation_samples = {x: y for x, y in list(dataset.items())[:max_samples]}
    return evaluation_samples


def build_samples_per_aspect(max_per_aspect=TARGET_PER_ASPECT):
    aspect_samples = {}

    for aspect_key in ASPECTS.keys():
        saved_indices_file = os.path.join(OUTPUT_DIR, f"{aspect_key}_samples.json")

        if os.path.exists(saved_indices_file):
            with open(saved_indices_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            aspect_samples[aspect_key] = saved_data

        else:
            evaluation_samples = load_medical_abstracts_for_evaluation()

            candidates = []
            for idx, sample_entry in evaluation_samples.items():
                prediction_text = sample_entry.get("abstract", "")

                if not prediction_text:
                    continue

                if aspect_key not in sample_entry:
                    continue

                sentences = sample_entry[aspect_key]
                non_na = [s for s in sentences if s and "not applicable" not in s.lower()]
                if len(non_na) == 0:
                    continue

                candidates.append({
                    "index": str(idx),
                    "abstract": prediction_text.strip(),
                    "meta": {
                        "n_summaries": len(non_na),
                        "sample_sentences": non_na[:2]
                    }
                })

            random.shuffle(candidates)
            selected_samples = candidates[:max_per_aspect]
            selected_sample_dict = {x["index"]: x for x in selected_samples}
            aspect_samples[aspect_key] = selected_sample_dict

            with open(saved_indices_file, "w", encoding="utf-8") as f:
                json.dump(selected_sample_dict, f, indent=2)

    return aspect_samples


def make_pair_prompt(aspect_title, aspect_instruction, rubric_lines,
                     abstract_a, abstract_b):
    rubric_text = "\n".join(f"{line}" for line in rubric_lines)
    prompt = (
        f"Abstract A:\n{abstract_a}\n\n"
        f"Abstract B:\n{abstract_b}\n\n"
        f"Task: For the aspect '{aspect_title}', judge how similar the two abstracts are on the scale defined below.\n\n"
        f"These are the detailed instructions for this aspect: {aspect_instruction}\n\n"
        f"Scoring rubric (use integers only):\n{rubric_text}\n\n"
        "Return EXACTLY a single JSON object and nothing else with the following format:\n"
        "{\"reasoning\": <a short summary of the differences and similarities, and which category this supports>, \"score\": <integer score>}\n\n"
        "Do not include any other text, commentary, or examples.\n"
    )
    return prompt


def assess_pair_with_llm(llm, model_params, user_prompt, max_tokens=MAX_TOKENS, retry_max=RETRY_MAX):
    messages = [
        {"role": "system", "content": "You are a scientific research assistant from the medical domain. "
                                      "You will be tasked with assessing the relatedness of two medical studies with regards to a specific aspect based on their abstracts. "
                                      "Note that both abstracts will for sure address medical research, which in itself therefore shall not be treated as indicator for relatedness. "
                                      "Provide concise, accurate responses and adhere precisely to the information that is actually present in the abstracts."},
        {"role": "user", "content": user_prompt}
    ]

    attempt = 0
    while attempt < retry_max:
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                **model_params
            )
            content = response["choices"][0]["message"]["content"]
            content = content.strip()
            parsed = safe_load_json_from_text(content)

            if "score" not in parsed:
                raise ValueError("JSON has no 'score' key.")

            score = int(parsed["score"])
            print(parsed)
            if score < 1 or score > 6:
                raise ValueError(f"score out of range: {score}")
            reason = parsed.get("reasoning", "")
            return {"score": score, "reason": reason, "raw": content}
        except Exception as e:
            attempt += 1
            wait = RETRY_BACKOFF * attempt
            print(f"[LLM ERROR] attempt {attempt}/{retry_max} -- {repr(e)} -- retrying in {wait}s")
            time.sleep(wait)

    return {"score": None, "reason": f"FAILED after {retry_max} attempts", "raw": ""}


def count_none_values_in_file(file_path):
    none_count = 0
    total_count = 0

    if not os.path.exists(file_path):
        return 0, 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                total_count += 1
                if obj.get("score") is None:
                    none_count += 1
            except Exception:
                continue

    return none_count, total_count


def get_none_pairs_from_file(file_path, samples):
    none_pairs = []

    if not os.path.exists(file_path):
        return none_pairs

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("score") is None:
                    a, b = obj["a"], obj["b"]

                    if a in samples and b in samples:
                        none_pairs.append((a, b))
            except Exception:
                continue

    return none_pairs


def update_jsonl_file(file_path, pair_key, new_score):
    if not os.path.exists(file_path):
        return

    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        try:
            obj = json.loads(line)
            a, b = obj["a"], obj["b"]
            if (min(a, b), max(a, b)) == pair_key:
                obj["score"] = new_score
                updated_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                updated_lines.append(line)
        except Exception:
            updated_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)


def process_aspect_pairs(llm, model_params, aspect_key, samples, out_dir=OUTPUT_DIR, save_every=SAVE_EVERY):
    out_file = os.path.join(out_dir, f"{aspect_key}_pairs.jsonl")

    idxs = list(samples.keys())
    total_pairs = len(idxs) * (len(idxs) - 1) // 2
    print(f"[RUN] aspect {aspect_key} -> {len(idxs)} samples -> {total_pairs} pairs")

    processed_pairs_with_valid_scores = set()
    none_pairs = []

    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    a, b = obj["a"], obj["b"]
                    pair_key = (min(a, b), max(a, b))

                    if obj.get("score") is not None:
                        processed_pairs_with_valid_scores.add(pair_key)
                    else:
                        if a in samples and b in samples:
                            none_pairs.append((a, b))
                except Exception:
                    continue

        print(f"[RESUME] Found {len(processed_pairs_with_valid_scores)} pairs with valid scores for {aspect_key}")
        print(f"[RETRY] Found {len(none_pairs)} pairs with None scores to retry for {aspect_key}")

    # Find completely missing pairs
    missing_pairs = [(a, b) for (a, b) in combinations(idxs, 2)
                     if (min(a, b), max(a, b)) not in processed_pairs_with_valid_scores]

    # Remove None pairs from missing pairs to avoid duplicates
    none_pair_keys = set((min(a, b), max(a, b)) for a, b in none_pairs)
    new_pairs = [(a, b) for (a, b) in missing_pairs
                 if (min(a, b), max(a, b)) not in none_pair_keys]

    pairs_to_process = none_pairs + new_pairs
    print(f"[TO PROCESS] {len(new_pairs)} new pairs + {len(none_pairs)} retry pairs = {len(pairs_to_process)} total pairs for {aspect_key}")

    if not pairs_to_process:
        print(f"[SKIP] No pairs to process for {aspect_key}")
        return

    f_out = open(out_file, "a", encoding="utf-8")

    progress = tqdm(total=len(pairs_to_process), desc=f"Aspect {aspect_key}", file=sys.stdout)
    processed_count = 0
    last_save = 0
    start_time = time.time()

    for (a, b) in pairs_to_process:
        aa = samples[a]["abstract"][:ABSTRACT_TRUNC_CHARS]
        bb = samples[b]["abstract"][:ABSTRACT_TRUNC_CHARS]

        user_prompt = make_pair_prompt(ASPECTS[aspect_key][0],
                                       ASPECTS[aspect_key][1],
                                       ASPECTS[aspect_key][2],
                                       aa, bb)

        result = assess_pair_with_llm(llm, model_params, user_prompt)

        pair_key = (min(a, b), max(a, b))

        if (a, b) in none_pairs:
            update_jsonl_file(out_file, pair_key, result.get("score"))
        else:
            out_obj = {
                "a": a,
                "b": b,
                        "score": result.get("score")
                    }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

        processed_count += 1
        last_save += 1
        progress.update(1)

        if last_save >= save_every:
            f_out.flush()
            os.fsync(f_out.fileno())
            last_save = 0
            elapsed = time.time() - start_time
            avg = elapsed / (processed_count if processed_count > 0 else 1)
            remain = (len(pairs_to_process) - processed_count) * avg

            hrs = int(remain // 3600)
            mins = int((remain % 3600) // 60)
            print(f"[SAVE] {processed_count}/{len(pairs_to_process)} done. Est remaining: {hrs}h {mins}m")

    f_out.flush()
    os.fsync(f_out.fileno())
    f_out.close()

    progress.close()
    print(f"[DONE] aspect {aspect_key}: processed {processed_count} pairs.")

    none_count, total_count = count_none_values_in_file(out_file)
    print(f"[NONE COUNT] aspect {aspect_key}: {none_count} None values out of {total_count} total pairs ({none_count/total_count*100:.1f}%)")


def create_all_assessments(run_aspects=None):
    if run_aspects is None:
        run_aspects = list(ASPECTS.keys())

    print("[BUILD] selecting samples per aspect...")
    aspect_samples = build_samples_per_aspect(max_per_aspect=TARGET_PER_ASPECT)

    print("[MODEL] loading Mistral LLM...")
    llm, model_params = load_llm()
    print("[MODEL] loaded.")

    for aspect_key in run_aspects:
        samples = aspect_samples.get(aspect_key, [])
        if len(samples) < 2:
            print(f"[SKIP] aspect {aspect_key} has <2 samples, skipping.")
            continue
        print(f"[START] Running aspect {aspect_key} with {len(samples)} samples.")
        process_aspect_pairs(llm, model_params, aspect_key, samples, out_dir=OUTPUT_DIR, save_every=SAVE_EVERY)

    print("[ALL DONE]")