import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from collections import defaultdict


# scaling of numeric columns
def scale_numeric(df, col):
    scaler = MinMaxScaler()
    df[col + '_scaled'] = scaler.fit_transform(df[[col]])
    df.drop([col], axis=1, inplace=True)
    return df

def apply_PCA(tensor, dim):
    pca_tensor = PCA(n_components=dim).fit(tensor)
    return torch.tensor(pca_tensor.transform(tensor)).float()


# encoding of short texts
# boost: apply PCA after SBERT on the individual attribute. Apply the mean to nan values
def encoding_short_text(df, col, target_dim=28):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # df[col+'_encoded'] = df.apply(lambda x: model.encode(x[col]), axis=1)

    # Convert the column to string if it's categorical
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        # if pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].astype(str)

    # Generate embeddings for text entries
    def get_embedding(text_or_list):
        ''' if pd.isna(text_or_list) or text_or_list is None:
            return None '''
        if isinstance(text_or_list, str):
            if text_or_list == "":
                return None
            return model.encode(text_or_list)
        if isinstance(text_or_list, list):
            embeddings = [model.encode(text) for text in text_or_list if text != ""]
            if embeddings:
                return np.mean(embeddings, axis=0)
            return None
        return None

    embeddings = df[col].apply(get_embedding)
    # non_nan_embeddings = embeddings.dropna().values
    non_nan_embeddings = [e for e in embeddings if e is not None]

    if len(non_nan_embeddings) > 0:
        embeddings_matrix = np.vstack(non_nan_embeddings)
        mean_embedding = np.mean(embeddings_matrix, axis=0)

        def replace_missing(embedding):
            if embedding is None:
                return mean_embedding
            else:
                return embedding

        embeddings = embeddings.apply(replace_missing)

    embeddings_matrix = np.vstack(embeddings)
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    df[col + '_encoded'] = [embedding.tolist() for embedding in
                            reduced_embeddings]  # Store the reduced embeddings in a single column as lists
    df.drop([col], axis=1, inplace=True)
    return df


def extract_meta_paths_DPSG_format(data):
    news_metapath_dict = defaultdict(set)  # Using a set to avoid duplicate entries

    for metapath in ["metapath_0", "metapath_1", "metapath_2"]:
        edge_index = data[("news", metapath, "news")].edge_index
        src_nodes, dst_nodes = edge_index[0].tolist(), edge_index[1].tolist()

        # Populate the dictionary
        for src, dst in zip(src_nodes, dst_nodes):
            news_metapath_dict[src].add(dst)  # Add dst to src's connection list
            news_metapath_dict[dst].add(src)  # Ensure bidirectionality

    # Convert sets to lists for final output
    news_metapath_dict = {k: list(v) for k, v in news_metapath_dict.items()}
    return news_metapath_dict