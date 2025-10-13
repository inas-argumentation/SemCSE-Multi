import torch
from transformers import AutoModel
from settings import proj_dimension

class UnifiedEmbeddingModel(torch.nn.Module):
    def __init__(self, base_model_name, prompt_identifiers, embedding_dim=proj_dimension):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.embedding_dim = embedding_dim
        self.prompt_identifiers = prompt_identifiers

        self.prompt_projections = torch.nn.ModuleDict({
            p: torch.nn.Linear(self.hidden_size, embedding_dim, bias=False)
            for p in self.prompt_identifiers
        })

        for module in self.prompt_projections.values():
            torch.nn.init.normal_(module.weight, mean=0.0, std=1e-2)

    def forward(self, input_ids, attention_mask, **kwargs):
        base_embedding = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:, 0]

        embeddings = {}
        for p in self.prompt_identifiers:
            embeddings[p] = self.prompt_projections[p](base_embedding)

        return embeddings

class MistralTransformationModel(torch.nn.Module):

    def __init__(self, embedding_dim, llm_hidden_size, num_prompt_tokens, num_mapped_tokens):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_mapped_tokens = num_mapped_tokens

        self.projection_net = torch.nn.Linear(embedding_dim, num_mapped_tokens * llm_hidden_size)

        self.embedding_to_hidden = torch.nn.Linear(embedding_dim, 768)
        self.activation = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.hidden_to_model_embedding = torch.nn.Linear(768, num_mapped_tokens * llm_hidden_size)

        self.prompt_embeddings = torch.nn.Parameter(
            torch.randn(1, num_prompt_tokens, llm_hidden_size, dtype=torch.float32)
        )
        torch.nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)

    def forward(self, embedding, llm_dtype):
        batch_size = embedding.shape[0]
        hidden_state = self.dropout(self.embedding_to_hidden(embedding))
        predicted_embeddings = self.hidden_to_model_embedding(hidden_state).view(
            batch_size, self.num_mapped_tokens, self.llm_hidden_size)
        expanded_prompts = self.prompt_embeddings.expand(batch_size, -1, -1)

        projected_embeddings = predicted_embeddings.to(llm_dtype)
        expanded_prompts = expanded_prompts.to(llm_dtype)

        return torch.cat([expanded_prompts, projected_embeddings], dim=1)

class LlamaTransformationModel(torch.nn.Module):

    def __init__(self, embedding_dim, llama_hidden_size, model_dtype, decoded_tokens=5, prefix_tokens=5, suffix_tokens=1):
        super().__init__()
        self.decoded_tokens = decoded_tokens
        self.model_dtype = model_dtype
        self.embedding_dim = embedding_dim
        self.llama_hidden_size = llama_hidden_size
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens

        self.embedding_to_hidden = torch.nn.Linear(embedding_dim, 768)
        self.activation = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.hidden_to_model_embedding = torch.nn.Linear(768, llama_hidden_size * decoded_tokens)

        self.prefix_embeddings = torch.nn.Parameter(torch.randn(prefix_tokens, llama_hidden_size) * 0.02)
        self.suffix_embeddings = torch.nn.Parameter(torch.randn(suffix_tokens, llama_hidden_size) * 0.02)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        hidden_states = self.dropout(self.activation(self.embedding_to_hidden(embeddings)))
        transformed_embeddings = self.hidden_to_model_embedding(hidden_states).reshape(batch_size, self.decoded_tokens, self.llama_hidden_size)

        prefix_batch = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_batch = self.suffix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        full_embeddings = torch.cat([prefix_batch, transformed_embeddings, suffix_batch ], dim=1)
        return full_embeddings.to(self.model_dtype)