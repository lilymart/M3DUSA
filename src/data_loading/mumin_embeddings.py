import os
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.utils import get_base_dir


def load_mumin_embeddings(emb_dir, emb_type, seed): #["only_text_news", "only_net_edges_cl", "only_net_edges_mps_cl", "only_net_edges_meta_cl", "only_net_edges_mps_meta_cl"]
    if emb_type == 'only_text_news':
        return load_claim_embeddings()
    if "_cl" in emb_type:
        emb_type = emb_type[:-3] #remove_cl
    return np.load(os.path.join(get_base_dir(), emb_dir, f'embeddings_{emb_type}_seed_{seed}.npy'))


def load_claim_embeddings():
    df = pd.read_parquet("/mnt/nas/pisani/dataset_mumin/claim_LP_split_60-15-25_with_embeddings.parquet")
    embeddings = np.stack(df['embedding'].values)
    return embeddings


def get_news_embedding_dim():
    df = pd.read_parquet("/mnt/nas/pisani/dataset_mumin/claim_LP_split_60-15-25_with_embeddings.parquet")
    first_embedding = df['embedding'].iloc[0] #already numpy array
    return first_embedding.shape[0]


def get_reduced_embedding(embedding, embedding_dim):
    embedding_reduced = PCA(n_components=embedding_dim).fit(embedding)
    return torch.tensor(embedding_reduced.transform(embedding)).float()
