import os
import json
import time
import random
import sys
from itertools import combinations
from collections import OrderedDict
from tqdm import tqdm
from llama_cpp import Llama
from invasion_biology_embeddings.auxiliary import relative_path, load_invasion_dataset

MODEL_REPO_ID = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
MODEL_FILENAME_GGUF = "*Q8_0.gguf"
N_CTX = 32000
TEMPERATURE = 0.8
MAX_TOKENS = 32000

TARGET_PER_ASPECT = 5
SAVE_EVERY = 10
ABSTRACT_TRUNC_CHARS = 1000
OUTPUT_DIR = relative_path("../data/llm_pairwise_assessments/invasion_biology")
INPUT_SUMMARIES_JSON = relative_path("../data/mistral_invasion_biology_summaries.json")
RANDOM_SEED = 0
RETRY_MAX = 3
RETRY_BACKOFF = 1

PAIR_START_IDX = None
PAIR_END_IDX = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

ASPECTS = OrderedDict([

    ("hypothesis", (
        "general relationship / hypothesis",
        (
            "Compare the high-level ecological relationship or directional hypothesis addressed in the two abstracts. "
            "Abstract away from species, place names, numeric values, and study-specific outcomes. "
            "Focus on whether the same broad cause–effect relationship is being studied. "
            "Assess similarity based on:\n"
            "(A) primary driver(s) / independent variable(s) (e.g., propagule pressure, disturbance, connectivity, enemy release, nutrient enrichment, climatic factor),\n"
            "(B) primary response(s) / dependent outcome(s) (e.g., establishment probability, abundance, spread rate, impact on native diversity, survival, recruitment), and\n"
            "(C) directional/causal framing or dominant mechanism (e.g., 'increases', 'reduces', 'facilitates', 'is associated with'; causal vs correlational framing; explicit mechanism like competition release or predation pressure).\n\n"
            "(D) specific methods of measuring the high-level variables (e.g., how is \"release from enemies\" or \"species diversity\" quantified).\n"
            "Important: Do NOT up-score similarity just because both papers are about biological invasions. This also means that many studies will measure \"invasion success\", "
            "which therefore shall not mean that they are related yet. For this specific variable, look at the more specific ways in which invasion success is measured/quantified, and at the remaining context of the high-level relationship. "
            "The default score is 1 unless there is a substantive match in some of the components. "
            "Minor overlaps (e.g., both about 'biotic interactions' without matching variables) should remain at 1 or 2."
        ),
        [
            "5: Same high-level relationship with matching details — the primary driver(s) AND primary response(s) clearly match, AND the directional/causal framing or mechanism is the same or equivalent AND the high-level variables are quantified in strongly related ways.",
            "4: Same high-level relationship without matching details — the primary driver(s) and primary response(s) clearly match (e.g., both evaluate if 'native species diversity decreases the likelihood of invasions'), but the scope is different and thus variables are quantified differently.",
            "3: Related — a strong match on ONE core component (e.g., both mention species diversity, disturbance, enemy release, etc.) with some additional similar factor (e.g., same driver linked to a somewhat similar but not identical response, or same response but different classes of driver).",
            "2: Weak relation — share only a minor conceptual element (e.g., both mention diversity, disturbance, or climate broadly) but the actual hypothesized driver–response link differs.",
            "1: No meaningful overlap in the high-level relationship/hypothesis (beyond both being invasion studies and targeting species invasions)."
        ]
    )),

    ("ecosystem", (
        "ecosystem / habitat context",
        (
            "Compare ONLY the ecosystem(s) or habitat context described in each abstract using broad, invasion-relevant ecological categories. "
            "Judge similarity using, for example, the following factors: "
            "(A) dominant habitat class and structural context (e.g., temperate grassland, freshwater lentic system, lotic river, estuarine/coastal, marine benthic/pelagic, urban green space), "
            "(B) abiotic regime and key gradients that shape invasibility (e.g., climate regime, salinity, hydrology/flow, substrate or soil type, disturbance frequency), and "
            "(C) invasion-relevant ecological features (e.g., native species richness, habitat fragmentation/connectivity, anthropogenic influence, presence of refugia or propagule sources). "
            "Use only habitat descriptors explicitly stated in the abstracts. Treat common synonyms and close subtypes as equivalent (e.g., 'lentic wetland' ≈ 'lake/pond'; 'coastal saltmarsh' ≈ 'tidal marsh'). "
            "If a study covers multiple habitat types or a landscape mosaic, assess overlap across the set of habitats (partial overlap should reduce the score).\n"
            "Note that the dominant habitat class is the most important factor. If it does not match (e.g., forest vs. grassland, freshwater vs. marine), then the score has to be 1 or 2.\n"
        ),
        [
            "6: A score of 6 is the highest score that shall be assigned to ecosystems that are equivalent.",
            "5: A score of 5 is a very high score that shall be assigned to very similar ecosystems."
            "This means that the dominant habitat class must match (e.g., both grassland or both forest) AND several other components (abiotic regime OR invasion-relevant features) match; The ONLY remaining differences are minor subtype or contextual contrasts.",
            "4: A score of 4 is a high score that shall be assigned to ecosystems that belong to the same ecological realm (grassland, forest, freshwater, coastal/marine, urban, etc.), "
            "but have at least one additional and notable similarity (e.g., both are subject to disturbance, both are the habitat of species with similar traits) or other important overlapping factor that is relevant to species invasions.",
            "3: A score of 3 is a medium score for ecosystems that are somewhat similar, which is mainly the case if they match with regards to their more precise class. "
            "Examples include cases in which both ecosystems are freshwater systems, or if both ecosystems are forest ecosystems, with no substantial additional overlap. "
            "If the classes do not match (for example, if one ecosystem is urban and the other is grasslands), then you will need to predict a lower class like 1 or 2.",
            "2: A score of 2 is a low score that shall be assigned to ecosystems that have almost no similarities. "
            "This is the case if they share only the broadest class, for example, if both are aquatic, but if there is no substantial similarity beyond that. "
            "1: A score of 1 is the lowest score that shall be assigned to completely different ecosystems. "
            "This is mainly the case for different habitat classes, for example, if one ecosystem is a grassland ecosystem and one is aquatic. "
            "The score of 1 is also the default category for an insufficient ecosystem description. \n"
            "Note that ALL abstracts will be about species invasions, so this alone is NOT a cause for a higher score and shall still be assessed with the score of 1. "
            "Additionally, note that the habitat type (forest, marine, urban, freshwater, grassland, etc.) must always match for the classes 3, 4, 5 and 6. "
            "If they do not match or are not specified sufficiently, the assessment has to be either 1 or 2."
        ]
    )),

    ("researchquestion", (
        "research question / central objective",
        (
            "Compare ONLY the central, most generalizable research question or comparative focus stated in both scientific abstracts. "
            "Judge similarity on the basis of core components when they are explicitly present:\n"
            "(A) the comparative contrast or experimental/observational focus (e.g., treatments vs controls, invaded vs uninvaded, gradients, temporal comparisons, manipulations),\n"
            "(B) the primary response or outcome measured (e.g., establishment probability, growth, survival, spread rate, recruitment, community composition), and\n"
            "(C) the context or scope that constrains inference (spatial/temporal scale, interacting factors or co-variates, multi-site vs single-site, seasonal vs multi-year). \n\n"
            "Note that these components are just examples. If none of these components apply, or additional components are more suitable (e.g., for comparing the similarity of two meta-reviews), then resort to your own judgement about the factors that shall contribute. "
            "Abstract away from specific species names, place names, and numeric values — assess on the high-level question structure and targets. "
            "Treat synonymous phrasing as equivalent. Use only what is explicitly stated in the abstracts; do not infer missing components. "
            "Do NOT up-score merely because both studies are about invasions, since all scientific abstracts that you shall assess will be about invasion biology."
        ),
        [
            "5: Essentially the same research question — e.g., the two abstracts state the same comparative focus/contrast AND the same primary response/outcome AND a compatible context/scope (or clearly equivalent phrasing across A+B+C).",
            "4: Very similar question — the same comparative focus AND the same primary response (A+B match), but differ in scope/scale or an interacting factor (C differs), OR the same response+context but a slightly different contrast (e.g., gradient vs categorical comparison) with clearly aligned aims.",
            "3: Related questions — they share one strong element (either the core contrast OR the primary response OR the specific contextual scope) and a weaker alignment in another element, producing a recognizable but not close match.",
            "2: Slightly related — a minor thematic link exists (e.g., both study effects of the same environmental driver or both measure similar outcomes) but the central aims/contrasts differ substantially.",
            "1: Unrelated or insufficient explicit information — no meaningful overlap in central question components, or one/both abstracts do not state a clear central question (do not rely on general invasion context as evidence of similarity, since ALL studies are about invasions)."
        ]
    )),

    ("species", (
        "species / focal organisms",
        (
            "Compare ONLY the focal organism(s) described in each abstract using the species names, but also functional information. "
            "When present, consider these components for similarity: "
            "(A) taxonomic level indicated (species, genus, family, higher), "
            "(B) life- or body-form (e.g., perennial herb, woody shrub, small benthic fish), "
            "(C) trophic role (herbivore, omnivore, predator, detritivore), "
            "(D) dispersal / reproductive mode (wind-dispersed seeds, planktonic larvae, clonal spread, broadcast spawner), "
            "and (E) broad biome/habitat or climate affinity (temperate estuary, freshwater lentic system, tropical forest, arid grassland, urban). "
            "Ignore incidental mentions of non-focal taxa, since we are only interested in the focal species. "
            "If abstracts describe multiple focal species or community-level analyses, evaluate overlap in the focal sets or in the functional/trait descriptors; if no clear overlap exists, assign the lowest score. "
            "Crucially: do NOT treat the fact that both papers address invasion biology or that both species are invasive as evidence of species similarity, since all abstracts that you will assess are about invasive species."
        ),
        [
            "5: Essentially the same focal organisms — the abstracts name the same species (explicit match) OR they provide equivalent functional descriptions across multiple components (taxonomic level, life-form, trophic role, dispersal, and habitat affinity).",
            "4: Very close taxonomic or functional match — same genus or a very close taxonomic group OR highly similar functional/trait profile (e.g., both perennial herbaceous plants with clonal spread and wind-dispersed seeds; or both small benthic predatory fishes with planktonic larvae), even if species differ.",
            "3: Same broad organismal class or functional group (e.g., both plants, both fishes, both terrestrial vertebrates) with at least one additional trait or affinity in common (trophic role, dispersal mode, or habitat), but clear taxonomic or trait differences remain.",
            "2: Weak similarity — share a single narrow trait or affinity (e.g., both are generalist predators) or only the same broad habitat class, but otherwise differ in type, core functional role, or taxonomic placement.",
            "1: No meaningful similarity in focal organisms — different taxa and functional roles, non-overlapping focal sets in community/multi-species studies, or insufficient/absent organismal description. Do NOT up-score based on both being invasive species."
        ]
    )),

    ("methodology", (
        "methodological structure",
        "Compare ONLY the methodological structure used by each study, mostly independent of species, ecosystems, variables measured, results, or management outcomes. \
    If applicable, take these factors into account:\n\
    (A) Study design class — e.g., manipulative/experimental (field, lab, mesocosm), observational/monitoring (surveys, transects, time series), modelling/simulation (statistical, correlative SDM, process-based), synthesis/meta-analysis.\n\
    (B) Data source/modality — e.g., replicated plots/transects, long-term monitoring records, lab/mesocosm trials, remote-sensing imagery, genetic data, literature datasets, simulation outputs.\n\
    (C) Precise method of data acquisition — e.g., sampling, trapping, literature survey, replicated trials, measurements, etc.\n\
    (D) Analysis framework — e.g., experimental contrasts/causal inference, comparative/cross-sectional, longitudinal/time-series, predictive modelling/SDM/ML, process-based simulation, meta-analysis/meta-regression.\n\
    (E) Scope — e.g., timeframe/duration, spatial extend, etc.\n\
    (F) Broad category of target species — e.g., plants, aquatic animals, mammals, etc.\n\
    Note that these factors might not apply to the studies given to you. Also note, that the factors differ in importance. For example,\
    if both studies address the same species with a similar analysis framework and scope, but one is observational and another is a modelling study, then they shall still be rather unrelated.\
    Note that the exact terminology might differ, but might still refer to the same underlying concept.\
    Use only what is explicitly stated in the abstracts.\
    Do not up-score similarity merely because both papers are about invasion biology.",
        [
            "6: Equivalent study design, like performing the exact same analysis on a different species with exact methodological equivalence.",
            "5: Extremely similar study design, matching most of the factors specified above.",
            "4: Very similar study design, matching many of the factors specified above.",
            "3: Same broad methodological family and similar study design, matching several of the factors specified above.",
            "2: Same broad methodological family with no or minimal additional similarities.",
            "1: No meaningful methodological overlap beyond both addressing invasion biology, or the abstracts lack sufficient methodological detail to justify similarity.",
            "Perform this assessment like an ecologist would, and judge the importance of each factor in the context of the abstracts yourself."
        ]
    )),

    ("recommendation", (
        "recommendations",
        (
            "Compare ONLY the explicit management or policy recommendations stated in each abstract. "
            "Evaluate similarity using, for example, the following factors, if applicable: "
            "(A) recommended action/instrument (what to do — e.g., early detection & rapid response, targeted removal, introduction of additional species, biological control, ballast-water treatment, pre-border screening, habitat restoration, monitoring, collaboration between organisations, creation of standards), "
            "(B) management objective/target outcome (why — e.g., prevent introduction, reduce spread, mitigate impacts, eradicate, restore), "
            "(C) implementation/governance scale or mechanism if stated (how/where — e.g., local eradication, landscape restoration, national policy, pre-border measures). "
            "Abstract away from species names, place names, numbers, and study-specific details. Treat close paraphrases and standard synonyms as matching. "
            "Use only the recommendations explicitly written in the abstracts; if an abstract contains no explicit management or policy recommendation, treat it as having no recommendation.",
        ),
        [
            "6: Essentially the same explicit recommendation — the two abstracts recommend the same action/instrument and have the same management objective (and, if stated, a compatible implementation scale).",
            "5: Same management objective/target outcome and very similar scale and very related concrete actions or instruments to achieve it (e.g., both propose strongly related practical interventions at the affected site, or both propose very related government regulation).",
            "4: Same management objective/target outcome and somewhat related concrete actions or instruments to achieve it (e.g., both recommend practical interventions at affected sites, but they are rather unrelated).",
            "3: Same management objective/target outcome and same broad management domain (prevention, control/eradication, restoration, monitoring/research, policy/regulation) but very different concrete actions or instruments to achieve it.",
            "2: Recommendations are vaguely related (e.g., just same objective, or just same broad management domain), but are otherwise different.",
            "1: No meaningful overlap in explicit recommendations, OR one or both abstracts contain no explicit management/policy recommendation."
        ]
    )),
])

def load_mistral_llm(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME_GGUF, n_ctx=N_CTX):
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
        candidate = text[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from model output.")


def build_samples_per_aspect(max_per_aspect=TARGET_PER_ASPECT):
    aspect_samples = {}

    for aspect_key in ASPECTS.keys():
        saved_indices_file = os.path.join(OUTPUT_DIR, f"{aspect_key}_samples.json")

        if os.path.exists(saved_indices_file):
            with open(saved_indices_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            aspect_samples[aspect_key] = saved_data

        else:
            dataset = load_invasion_dataset()
            with open(INPUT_SUMMARIES_JSON, "r") as f:
                summary_dataset = json.load(f)

            candidates = []
            for key_str, sample_entry in dataset["labeled"].items():
                idx = int(key_str)
                prediction_text = sample_entry["prediction_text"]
                summaries = summary_dataset["labeled"][key_str]

                if len([x for x in summaries if aspect_key in x]) == 0:
                    continue

                sentences = [x for y in [summaries[a] for a in summaries if aspect_key in a] for x in y]
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
        {"role": "system", "content": "You are a scientific research assistant from the field of invasion biology. "
                                      "You will be tasked with assessing the relatedness of two scientific studies from the field of invasion biology with regards to a specific aspect based on their abstracts. "
                                      "Note that both abstracts will for sure address the field of invasion biology, which in itself therefore shall not be treated as indicator for relatedness. "
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
            reason = parsed.get("reason", "")
            return {"score": score, "reason": reason, "raw": content}
        except Exception as e:
            attempt += 1
            wait = RETRY_BACKOFF * attempt
            print(f"[LLM ERROR] attempt {attempt}/{retry_max} -- {repr(e)} -- retrying in {wait}s")
            time.sleep(wait)

    return {"score": None, "reason": f"FAILED after {retry_max} attempts", "raw": ""}

def process_aspect_pairs(llm, model_params, aspect_key, samples, out_dir=OUTPUT_DIR, save_every=SAVE_EVERY):
    out_file = os.path.join(out_dir, f"{aspect_key}_pairs.jsonl")

    idxs = list(samples.keys())
    all_pairs = list(combinations(idxs, 2))
    total_pairs = len(all_pairs)

    print(f"[RUN] aspect {aspect_key} -> {len(idxs)} samples -> {total_pairs} total pairs")

    processed_pairs = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    a,b = obj["a"], obj["b"]
                    processed_pairs.add((min(a,b), max(a,b)))
                except Exception:
                    continue
        print(f"[RESUME] Found {len(processed_pairs)} already processed pairs")

    pairs_to_process = [(a,b) for (a,b) in all_pairs if (min(a,b),max(a,b)) not in processed_pairs]
    print(f"[TO PROCESS] {len(pairs_to_process)} pairs remain for {aspect_key}")

    f_out = open(out_file, "a", encoding="utf-8")

    progress = tqdm(total=len(pairs_to_process), desc=f"Aspect {aspect_key}", file=sys.stdout)
    processed_count = 0
    last_save = 0
    start_time = time.time()

    for (a,b) in pairs_to_process:
        aa = samples[a]["abstract"][:ABSTRACT_TRUNC_CHARS]
        bb = samples[b]["abstract"][:ABSTRACT_TRUNC_CHARS]

        user_prompt = make_pair_prompt(ASPECTS[aspect_key][0],
                                      ASPECTS[aspect_key][1],
                                      ASPECTS[aspect_key][2],
                                      aa, bb)

        result = assess_pair_with_llm(llm, model_params, user_prompt)
        out_obj = {
            "a": a,
            "b": b,
            "score": result.get("score")}
        f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        processed_count += 1
        last_save += 1
        progress.update(1)

        if last_save >= save_every:
            f_out.flush()
            os.fsync(f_out.fileno())
            last_save = 0
            elapsed = time.time() - start_time
            avg = elapsed / (processed_count if processed_count>0 else 1)
            remain = (len(pairs_to_process) - processed_count) * avg

            hrs = int(remain//3600); mins = int((remain%3600)//60)
            print(f"[SAVE] {processed_count}/{len(pairs_to_process)} done. Est remaining: {hrs}h {mins}m")

    f_out.flush()
    os.fsync(f_out.fileno())
    f_out.close()
    progress.close()
    print(f"[DONE] aspect {aspect_key}: processed {processed_count} pairs.")

def create_all_assessments():
    print("[BUILD] selecting samples per aspect...")
    aspect_samples = build_samples_per_aspect(max_per_aspect=TARGET_PER_ASPECT)

    print("[MODEL] loading Mistral LLM...")
    llm, model_params = load_mistral_llm()
    print("[MODEL] loaded.")

    for aspect_key in list(ASPECTS.keys()):
        samples = aspect_samples.get(aspect_key, [])
        if len(samples) < 2:
            print(f"[SKIP] aspect {aspect_key} has <2 samples, skipping.")
            continue
        print(f"[START] Running aspect {aspect_key} with {len(samples)} samples.")
        process_aspect_pairs(llm, model_params, aspect_key, samples, out_dir=OUTPUT_DIR, save_every=SAVE_EVERY)

    print("[ALL DONE]")