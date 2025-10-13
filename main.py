import os
if os.path.exists("/mnt/67FA8D9E50BFBFCF/huggingface"):
    os.environ['HF_HOME'] = "/mnt/67FA8D9E50BFBFCF/huggingface"

if os.path.exists("/mnt/data/mbrinner/huggingface"):
    os.environ['HF_HOME'] = "/mnt/data/mbrinner/huggingface"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def generate_invasion_biology_data():
    from invasion_biology_embeddings import generate_aspect_specific_summaries, generate_pairwise_assessment_dataset

    # Generate summaries for training
    generate_aspect_specific_summaries.generate_sentences_for_dataset()

    # Generate pairwise assessments for evaluation
    generate_pairwise_assessment_dataset.create_all_assessments()

def train_invasion_biology_models():
    from invasion_biology_embeddings import train_aspect_specific_embedding_models, train_unified_model, train_llama_decoders, train_mistral_decoders

    # Train individual embedding models
    train_aspect_specific_embedding_models.train_all_individual_prompt_embedding_models()

    # Distill unified model
    train_unified_model.train_unified_model()

    # Train decoder models
    train_llama_decoders.train_all_llama_decoders()
    train_mistral_decoders.train_all_mistral_decoders()

def evaluate_invasion_biology_models():
    from invasion_biology_embeddings import evaluate_individual_embedding_models, evaluate_unified_embedding_model, evaluate_decoders

    # Evaluate individual embedding models and the SemCSE baseline on the retrieval task
    evaluate_individual_embedding_models.main_evaluation()
    evaluate_individual_embedding_models.evaluate_baseline_semcse()

    # Evaluate unified model and baselines on the ground-truth correlation analysis
    evaluate_unified_embedding_model.main()
    evaluate_unified_embedding_model.evaluate_baseline_models()

    # Evaluate decoders on 1) the normal embedding decoding task, 2) the shuffled, non-matching decoding ablation,
    # 3) the unconditioned (embedding-free) ablation, and 4) the t-SNE reconstruction ablation.
    evaluate_decoders.evaluate_all_llama_decoders()
    evaluate_decoders.evaluate_all_llama_decoders()
    evaluate_decoders.evaluate_llama_decoders_shuffled()
    evaluate_decoders.evaluate_mistral_decoders_shuffled()
    evaluate_decoders.evaluate_models_unconditioned()
    evaluate_decoders.evaluate_tsne_roundtrip("llama")
    evaluate_decoders.evaluate_tsne_roundtrip("mistral")

def generate_medical_data():
    from medical_embeddings import generate_aspect_specific_summaries, generate_pairwise_assessment_dataset

    # Generate summaries for training
    generate_aspect_specific_summaries.generate_sentences_for_dataset()

    # Generate pairwise assessments for evaluation
    generate_pairwise_assessment_dataset.create_all_assessments()

def train_medical_models():
    from medical_embeddings import train_aspect_specific_embedding_models, train_unified_model, \
        train_llama_decoders, train_mistral_decoders

    # Train individual embedding models
    train_aspect_specific_embedding_models.train_all_individual_prompt_embedding_models()

    # Distill unified model
    train_unified_model.train_unified_model()

    # Train decoder models
    train_llama_decoders.train_all_llama_decoders()
    train_mistral_decoders.train_all_mistral_decoders()

def evaluate_medical_models():
    from medical_embeddings import evaluate_individual_embedding_models, evaluate_unified_embedding_model, \
        evaluate_decoders

    # Evaluate individual embedding models and the SemCSE baseline on the retrieval task
    evaluate_individual_embedding_models.main_evaluation()
    evaluate_individual_embedding_models.evaluate_baseline_semcse()

    # Evaluate unified model and baselines on the ground-truth correlation analysis
    evaluate_unified_embedding_model.main()
    evaluate_unified_embedding_model.evaluate_baseline_models()

    # Evaluate decoders on 1) the normal embedding decoding task, 2) the shuffled, non-matching decoding ablation,
    # 3) the unconditioned (embedding-free) ablation, and 4) the t-SNE reconstruction ablation.
    evaluate_decoders.evaluate_all_llama_decoders()
    evaluate_decoders.evaluate_all_llama_decoders()
    evaluate_decoders.evaluate_llama_decoders_shuffled()
    evaluate_decoders.evaluate_mistral_decoders_shuffled()
    evaluate_decoders.evaluate_models_unconditioned()
    evaluate_decoders.evaluate_tsne_roundtrip("llama")
    evaluate_decoders.evaluate_tsne_roundtrip("mistral")


if __name__ == '__main__':
    generate_invasion_biology_data()
    train_invasion_biology_models()
    evaluate_medical_models()

    generate_medical_data()
    train_medical_models()
    evaluate_medical_models()