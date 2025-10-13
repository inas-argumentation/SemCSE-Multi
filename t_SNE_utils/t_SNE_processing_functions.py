import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold import _utils
from t_SNE_utils.binary_search_perplexity import _binary_search_perplexity
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import cma

MACHINE_EPSILON = np.finfo(np.double).eps

def to_np(self):
    return self.detach().cpu().numpy()

setattr(torch.Tensor, "np", to_np)

def calculate_t_SNE_embeddings(embeddings):

    cosine_similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    t_sne_distances = (1 - cosine_similarities).cpu().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, metric='precomputed', init="random")
    embedded_texts = tsne.fit_transform(np.clip(t_sne_distances, a_min=0, a_max=2))
    return embedded_texts, cosine_similarities

class PCAMapper:
    def __init__(self, data, n_components=15):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(data)

        self.components = torch.tensor(self.pca.components_, dtype=torch.float32, device="cuda")
        self.mean = torch.tensor(self.pca.mean_, dtype=torch.float32, device="cuda")

    def transform(self, data):
        return self.pca.transform(data)

    def transform_pt(self, data):
        centered_data = data - self.mean
        return torch.matmul(centered_data, self.components.T)

    def inverse_transform(self, reduced_data):
        return self.pca.inverse_transform(reduced_data)

    def inverse_transform_pt(self, reduced_data):
        reconstructed = torch.matmul(reduced_data, self.components)
        return reconstructed + self.mean

    def get_components(self):
        return self.pca.components_

    def get_explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_

def create_distance_matrix(distances, embeddings, parameter, pca, inverse_pca):
    if inverse_pca:
        embedding = pca.inverse_transform_pt(parameter)
    else:
        embedding = parameter

    similarity = torch.nn.functional.cosine_similarity(embeddings, embedding, dim=-1)
    distance = (1-similarity)

    new_distances = torch.concatenate([distances, distance.unsqueeze(0)], dim=0)
    new_distances = torch.concatenate([new_distances, torch.concatenate([distance, torch.zeros(1, device="cuda", dtype=torch.float32)], dim=0).unsqueeze(1)], dim=1)
    return torch.clip(new_distances, 0, 2), embedding

def calc_P_pytorch(distances, mask, betas_torch=None):
    distances_np = torch.clone(distances).np()
    if betas_torch is None:
        conditional_P, betas = _binary_search_perplexity(distances_np, 30)
        betas_torch = torch.tensor(betas, dtype=distances.dtype, device=distances.device).view(-1, 1)
    P = torch.exp(-distances * betas_torch)

    P = P * mask

    row_sums = P.sum(dim=1, keepdim=True)
    row_sums = row_sums + 1e-8
    P = P / row_sums

    P = P + P.t()
    sum_P = torch.sum(P)
    P = P / sum_P
    return P, betas_torch

# Optimize a new high-dimensional embedding for a new low-dimensional point in the visualization
def optimize_embedding_for_t_SNE(embeddings, t_SNE_embeddings, target_point):
    pca = PCAMapper(embeddings.np(), n_components=20)

    target_point = np.expand_dims(target_point, axis=0)
    cosine_similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    t_sne_distances = (1 - cosine_similarities)
    all_t_SNE_embeddings = torch.tensor(np.concatenate([t_SNE_embeddings, target_point], axis=0), device="cuda", dtype=torch.float32)

    closest_indices = np.argsort(np.sum(np.square(t_SNE_embeddings - target_point), axis=-1))
    initial_embedding = embeddings[closest_indices[:5]].mean(0, keepdim=True)
    initial_embedding_pca = pca.transform_pt(initial_embedding)
    parameter = torch.tensor(initial_embedding_pca, requires_grad=True)

    optimizer = torch.optim.SGD([parameter], lr=1e-3, momentum=0.3)

    diagonal_mask = torch.eye(t_sne_distances.shape[0]+1, dtype=torch.float32, device="cuda")
    reverse_diagonal_mask = 1-diagonal_mask
    multiplier = all_t_SNE_embeddings.shape[0]

    X_embedded = all_t_SNE_embeddings.np()
    dist = pdist(X_embedded, "sqeuclidean")
    dist += 1.0
    dist **= -1
    Q = torch.tensor(squareform(np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)), device="cuda", dtype=torch.float32) * multiplier

    ema = None
    best_loss = None
    best_embedding_sgd = None
    betas_torch = None
    for iteration in range(10000):
        distance_matrix, embedding = create_distance_matrix(t_sne_distances, embeddings, parameter, pca, True)
        P, betas_torch = calc_P_pytorch(distance_matrix, reverse_diagonal_mask, None if iteration%10 == 0 else betas_torch)
        P = P * multiplier

        kl_divergence = torch.sum(P * torch.log((P + diagonal_mask) / (Q + diagonal_mask)))
        loss = kl_divergence
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if ema is None: ema = kl_divergence.item()
        else: ema = ema * 0.95 + kl_divergence.item() * 0.05

        if best_loss is None or best_loss > kl_divergence.item():
            best_loss = kl_divergence.item()
            best_embedding_sgd = torch.clone(embedding)

        if ema < kl_divergence.item() and iteration > 100: break

    return best_embedding_sgd

def calculate_Q_pytorch(t_sne_embedding_distances, t_sne_embeddings, parameter, mask):
    new_distances = torch.square(t_sne_embeddings - parameter.unsqueeze(0)).sum(-1)
    new_distance_matrix = torch.concatenate([t_sne_embedding_distances, new_distances.unsqueeze(0)], dim=0)
    new_distance_matrix = torch.concatenate([new_distance_matrix, torch.concatenate([new_distances, torch.zeros(1, device="cuda", dtype=torch.float32)], dim=0).unsqueeze(1)], dim=1)

    Q = 1 / (1 + new_distance_matrix)
    Q = Q * mask
    Q = Q / torch.sum(Q)

    return Q

# Embed an additional point into an existing t-SNE visualization
def embed_new_point(embeddings, previous_t_SNE_embeddings, new_sample_embedding, cosine_similarities=None):
    all_embeddings = torch.concatenate([embeddings, new_sample_embedding], dim=0)

    if cosine_similarities is None:
        cosine_similarities = torch.nn.functional.cosine_similarity(all_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=-1)

    t_sne_distances = (1 - cosine_similarities)

    t_SNE_embeddings = torch.tensor(previous_t_SNE_embeddings, device="cuda", dtype=torch.float32)
    t_sne_embedding_distances = torch.sum(torch.square(t_SNE_embeddings.unsqueeze(1) - t_SNE_embeddings), dim=2)

    closest_embeddings = torch.argsort(cosine_similarities[-1, :-1])
    initial_embedding = t_SNE_embeddings[closest_embeddings[-5:]].mean(0)
    parameter = torch.tensor(initial_embedding.clone().detach(), requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=1e-4, momentum=0.3)

    diagonal_mask = torch.eye(t_sne_distances.shape[0], dtype=torch.float32, device="cuda")
    reverse_diagonal_mask = 1 - diagonal_mask
    multiplier = all_embeddings.shape[0]

    distances = t_sne_distances.np()
    conditional_P = _utils._binary_search_perplexity(distances, 30, False)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = torch.tensor(P / sum_P, device="cuda", dtype=torch.float32) * multiplier

    ema = None
    best_loss = None
    best_param = None
    for iteration in range(1000):
        Q = calculate_Q_pytorch(t_sne_embedding_distances, t_SNE_embeddings, parameter, reverse_diagonal_mask) * multiplier
        kl_divergence = torch.sum(P * torch.log((P + diagonal_mask) / (Q + diagonal_mask)))

        loss = kl_divergence
        loss.backward()
        optimizer.step()

        if ema is None: ema = kl_divergence.item()
        else: ema = ema * 0.95 + kl_divergence.item() * 0.05

        if best_loss is None or best_loss > kl_divergence.item():
            best_loss = kl_divergence.item()
            best_param = torch.clone(parameter)

        if ema < kl_divergence.item() and iteration > 100: break

    def objective_function(x):
        param = torch.tensor(x, device="cuda", dtype=torch.float32)
        Q = calculate_Q_pytorch(t_sne_embedding_distances, t_SNE_embeddings, param, reverse_diagonal_mask) * multiplier
        kl_divergence = torch.sum(P * torch.log((P + diagonal_mask) / (Q + diagonal_mask)))
        return kl_divergence.item()

    es = cma.CMAEvolutionStrategy(best_param.np(), 0.2)

    for _ in range(50):
        solutions = es.ask()
        fitnesses = [objective_function(x) for x in solutions]
        es.tell(solutions, fitnesses)

    best_solution = es.result.xbest
    return best_solution