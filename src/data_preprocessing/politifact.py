import os
import pandas as pd
import numpy as np
import torch
import scipy
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import hdbscan  # from paper "Accelerated Hierarchical Density Based Clustering"
import re

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.transforms import AddMetaPaths
from torch_geometric.utils import index_to_mask, mask_to_index

from src.data_preprocessing.utils import apply_PCA
from src.utils import save_to_pickle, set_random_seed, training_seed
from src.utils import get_device

from src.utils import get_base_dir

import emoji

dir_base = '/mnt/nas/guarascio/fakenews_datasets/Politifact/politifact_in_mumin_format/'
output_dir = "/mnt/nas/martirano/politifact_cleaned"
target = "news"


# dir_base = '/mnt/nas/guarascio/fakenews_datasets/Politifact/politifact_in_mumin_format/'
# output_dir = "/home/scala/projects/GNN_ContinualLerning/data/politifact/"

def count_nan_or_empty(series):
    # return series.apply(lambda x: pd.isna(x) or (isinstance(x, list) and len(x) == 0)).sum()
    return series.apply(lambda x: pd.isna(x) or (isinstance(x, str) and (x == '[]' or x == 'none'))).sum()


def drop_columns_with_many_nans(df):
    threshold = len(df) / 2
    columns_to_drop = [col for col in df.columns if count_nan_or_empty(df[col]) > threshold]
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned


def drop_columns_starting_with(df, prefix):
    columns_to_drop = [col for col in df.columns if col.startswith(prefix)]
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned


def extract_base_source_news(url):
    if url is None:
        return "unknown"
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    # Extract second-level domain (e.g., cnn, nytimes)
    if len(domain_parts) >= 2:
        return domain_parts[-2]
    return "unknown"


def extract_base_source_tweet(source):
    match = re.search(r'>(.*?)<', source)
    return match.group(1) if match else "Unknown"


def label_encoding(df, col):
    label_encoder = LabelEncoder()
    encoded_data = label_encoder.fit_transform(df[col])
    # original_data = label_encoder.inverse_transform(encoded_data)
    # print(original_data)
    return encoded_data


# scaling of numeric columns
def scale_numeric(df, col):
    scaler = MinMaxScaler()
    df[col + '_scaled'] = scaler.fit_transform(df[[col]])
    df.drop([col], axis=1, inplace=True)
    return df


def convert_to_tensor(s):
    # array = ast.literal_eval(array_string)
    # return torch.tensor(array, dtype=torch.float32)
    arr = np.fromstring(s.strip('[]'), sep=' ')
    return torch.tensor(arr, dtype=torch.float32)


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


def clustering(df, col):
    threshold = 0.75
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=1 - threshold)
    labels = clusterer.fit_predict(df[col].tolist())
    df['cluster_label'] = labels
    return df


def outliers_remapping(df, col):
    new_values = []
    new_value_counter = df[col].max() + 1
    for value in df[col]:
        if value == -1:
            new_values.append(new_value_counter)
            new_value_counter += 1
        else:
            new_values.append(value)

    df[col + '_no_outliers'] = new_values
    return df


def clean_text(text):
    text = re.sub('RT @[A-Za-z0-9_]+:', '', text)  # retweet
    text = re.sub('@[A-Za-z0-9_]+', '', text)  # tag - users
    text = re.sub('https?://\S+|www\.\S+', '', text)  # url
    text = re.sub('#', '', text)  # hashtags
    # text = emoji.replace_emoji(text, replace='') #emoji
    text = emoji.demojize(text)
    return text


""" OLD 
def safe_tensor_conversion(x, dim):
    if isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float32)
    if isinstance(x, str):
        x = np.fromstring(x.strip('[]'), sep=' ')
        return torch.tensor(x, dtype=torch.float32)
    if pd.isna(x):
        return torch.zeros(dim, dtype=torch.float32)
    else:
        return torch.zeros(dim, dtype=torch.float32)
"""

""" NEW """


def safe_tensor_conversion(x, dim):
    if isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float32)
    if isinstance(x, str):
        x = np.fromstring(x.strip('[]'), sep=' ')
        return torch.tensor(x, dtype=torch.float32)
    if isinstance(x, np.ndarray):
        # Handle NaN arrays explicitly
        if np.isnan(x).all():  # Check if all elements are NaN
            return torch.zeros(dim, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32)
    if isinstance(x, list):
        x = np.array(x, dtype=np.float32)
        if np.isnan(x).all():
            return torch.zeros(dim, dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32)
    if pd.isna(x):  # Check for scalar NaN
        return torch.zeros(dim, dtype=torch.float32)
    else:
        return torch.zeros(dim, dtype=torch.float32)


def encoding_attributes(df):
    tensors = []
    for col in df.columns:
        print('##### Processing column ', col, ' #####')
        if df[col].dtype == 'int64':
            tensors.append(torch.tensor(df[col].values, dtype=torch.int32).unsqueeze(1))
        elif df[col].dtype == 'float64':
            tensors.append(torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1))
        elif df[col].dtype == 'bool':
            tensors.append(torch.tensor(df[col].values, dtype=torch.bool).unsqueeze(1))
        elif df[col].dtype == 'object':  # per le colonne con gli embedding
            first_element = df[col].dropna().iloc[0]
            dim = len(first_element) if isinstance(first_element, list) else \
            np.fromstring(first_element.strip('[]'), sep=' ').shape[0]  # str
            embedding_tensors = df[col].apply(
                lambda x: safe_tensor_conversion(x, dim) if not isinstance(x, torch.Tensor) else x)
            embedding_stack = torch.stack(embedding_tensors.tolist())  # Convert to list before stacking
            tensors.append(embedding_stack)
        elif df[col].dtype == 'category':
            enc = LabelEncoder()
            encoded_values = enc.fit_transform(df[col])
            tensors.append(torch.tensor(encoded_values, dtype=torch.int32).unsqueeze(1))
        else:
            raise ValueError(f"Unsupported column type: {df[col].dtype}")

    # Concatenate all tensors along the last dimension
    return torch.cat(tensors, dim=1)


def edges_encoding(df):
    return torch.tensor(df.values.T)


def edges_rev_encoding(df):
    df_rev = df[['tgt', 'src']]
    return torch.tensor(df_rev.values.T)




""" NEWS """
'''
df_N = pd.read_csv(os.path.join(dir_base, 'news_politifact.csv'))
df_N_ok = drop_columns_with_many_nans(df_N)
df_N_ok = drop_columns_starting_with(df_N_ok, 'meta_data')
#df_N_ok['base_source'] = df_N_ok["source"].apply(lambda x: extract_base_source_news(x) if pd.notna(x) else "unknown")
#df_N_ok.drop(columns=["images", "top_img", "url", "canonical_link"], index=1, inplace=True)
df_N_ok.to_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"), index=False) #NEW: text, title, news_id, label, base_source, embedding
'''

'''
print("processing NEWS nodes")
df_N = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"))
#print("Encoding text...")
#df_N = encoding_short_text(df_N, "text", target_dim=256)
print("Encoding title...")
df_N = encoding_short_text(df_N, "title", target_dim=128)
print("Encoding source...")
df_N = encoding_short_text(df_N, "base_source", target_dim=64)
NX = encoding_attributes(df_N[['title_encoded', 'embedding', 'base_source_encoded']].copy()) #'text_encoded'
print(NX.shape)
torch.save(NX, os.path.join(output_dir, "heterodata", "features", "NX_tensor.pt"))
N_labels = label_encoding(df_N, 'label').tolist()
NY = torch.tensor(N_labels)
torch.save(NY, os.path.join(output_dir, "heterodata", "NY_tensor.pt"))
'''
#processing only_meta
'''
df_N = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"))
print("Encoding title...")
df_N = encoding_short_text(df_N, "title", target_dim=128)
print("Encoding source...")
df_N = encoding_short_text(df_N, "base_source", target_dim=64)
NX = encoding_attributes(df_N[['title_encoded', 'base_source_encoded']].copy())
print(NX.shape)
torch.save(NX, os.path.join(output_dir, "heterodata", "features", "NX_only_meta_tensor.pt"))
'''
#processing only_text
'''
df_N = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"))
NX = encoding_attributes(df_N[['embedding']].copy())
print(NX.shape)
torch.save(NX, os.path.join(output_dir, "heterodata", "features", "NX_only_text_tensor.pt"))
'''


""" USER """
'''
df_U = pd.read_csv(os.path.join(dir_base, 'user_with_timeline_embedding_politifact.csv'))#df_U = pd.read_csv(os.path.join(dir_base, 'user_politifact.csv'))
df_U2 = drop_columns_with_many_nans(df_U)
df_U3 = df_U2.drop(columns=["is_translation_enabled", "screen_name"])
df_U3.to_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), index=False) #NEW: user_id, location, description, protected, followers_count, friends_count, listed_count,
print(df_U3.shape)
#listed_count, favourites_count, created_at, verified, statuses_count, has_extended_profile, default_profile, default_profile_image,
'''
'''
print("Processing USER nodes")
df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
print("Encoding location...")
df_U = encoding_short_text(df_U, "location", target_dim=128)
print("Encoding description...")
df_U = encoding_short_text(df_U, "description", target_dim=128)
df_U = scale_numeric(df_U, 'followers_count')
df_U = scale_numeric(df_U, 'friends_count')
df_U = scale_numeric(df_U, 'listed_count')
df_U = scale_numeric(df_U, 'favourites_count')
df_U = scale_numeric(df_U, 'statuses_count')
df_U = scale_numeric(df_U, 'created_at')
df_U_ok = df_U.drop(columns=["user_id"])
UX = encoding_attributes(df_U_ok)
print(UX.shape)
torch.save(UX, os.path.join(output_dir, "heterodata", "features", "UX_tensor.pt"))
'''

#processing only_meta
'''
df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
df_U = scale_numeric(df_U, 'followers_count')
df_U = scale_numeric(df_U, 'friends_count')
df_U = scale_numeric(df_U, 'listed_count')
df_U = scale_numeric(df_U, 'favourites_count')
df_U = scale_numeric(df_U, 'statuses_count')
df_U = scale_numeric(df_U, 'created_at')
df_U_ok = df_U.drop(columns=["user_id", "location", "description", "embedding"])
UX = encoding_attributes(df_U_ok)
print(UX.shape)
torch.save(UX, os.path.join(output_dir, "heterodata", "features", "UX_only_meta_tensor.pt"))
'''
#processing only_text
'''
df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
print("Encoding location...")
df_U = encoding_short_text(df_U, "location", target_dim=128)
print("Encoding description...")
df_U = encoding_short_text(df_U, "description", target_dim=128)
df_U_ok = df_U[['location_encoded', 'description_encoded']].copy()
UX = encoding_attributes(df_U_ok)
print(UX.shape)
torch.save(UX, os.path.join(output_dir, "heterodata", "features", "UX_only_text_tensor.pt"))
'''
#processing timeline
'''
df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
df_U_ok = df_U[['embedding']].copy()
UX = encoding_attributes(df_U_ok)
print(UX.shape)
torch.save(UX, os.path.join(output_dir, "heterodata", "features", "UX_only_timeline_tensor.pt"))
'''

""" HASHTAG """
'''
df_H = pd.read_csv(os.path.join(dir_base, 'hashtag_politifact.csv'))
#df_H.drop(columns=["hashtag_id"], index=1, inplace=True)
print(df_H.shape)
df_H.to_csv(os.path.join(output_dir, "original_data", "nodes", "hashtag.csv"), index=False)
'''
'''
df_H = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "hashtag.csv"))
print("Encoding texts...")
df_H = encoding_short_text(df_H, "hashtag", target_dim=256)
print("Performing clustering...")
df_H = clustering(df_H, "hashtag_encoded")
num_clusters = df_H['cluster_label'].unique().shape[0]-1
num_outliers = df_H[df_H['cluster_label']== -1].shape[0]
print(f"Number of clusters: {num_clusters}")
print(f"Number of outliers: {num_outliers}")
df_H = outliers_remapping(df_H, "cluster_label")
HX = encoding_attributes(df_H[['hashtag_encoded', 'cluster_label_no_outliers']].copy())
print(HX.shape)
torch.save(HX, os.path.join(output_dir, "heterodata", "features", "HX_tensor.pt"))
'''
# copy
'''
HX = torch.load(os.path.join(output_dir, "heterodata", "features", "HX_tensor.pt"))
print(HX.shape)
torch.save(HX, os.path.join(output_dir, "heterodata", "features", "HX_only_meta_tensor.pt"))
'''

""" TWEET """
'''
df_T = pd.read_csv(os.path.join(dir_base, 'tweet_politifact.csv'))
df_T = drop_columns_with_many_nans(df_T)
df_T_bot = pd.read_csv(os.path.join("/mnt/nas/scala", "tweet_embedding_politifact_with_bot.csv"))[['tweet_id', 'bot_generated']]
df_T_ok = df_T.merge(df_T_bot, on="tweet_id", how="left")
df_T_ok.drop(columns=["source"], axis=1, inplace=True)
print(df_T_ok.shape)
df_T_ok.to_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"), index=False)
'''
'''
print("Processing TWEET nodes")

df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"))
print(df_T.shape)
df_T.drop(columns = ["text", "tweet_id"], axis=1, inplace=True)
#df_T = encoding_short_text(df_T, "text", target_dim=384)
df_T['lang'] = df_T['lang'].astype('category')
df_T = scale_numeric(df_T, 'created_at')
df_T = scale_numeric(df_T, 'retweet_count')
df_T = scale_numeric(df_T, 'favorite_count')
TX = encoding_attributes(df_T)
print(TX.shape)
torch.save(TX, os.path.join(output_dir, "heterodata", "features", "TX_tensor.pt"))
'''

#processing only_meta
'''
df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"))
df_T.drop(columns = ["text", "tweet_id"], axis=1, inplace=True)
df_T['lang'] = df_T['lang'].astype('category')
df_T = scale_numeric(df_T, 'created_at')
df_T = scale_numeric(df_T, 'retweet_count')
df_T = scale_numeric(df_T, 'favorite_count')
TX = encoding_attributes(df_T)
print(TX.shape)
torch.save(TX, os.path.join(output_dir, "heterodata", "features", "TX_only_meta_tensor.pt"))
'''
#processing only_text
'''
df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"))
df_T_ok = df_T[['text_emb_twhin_bert_base']].copy()
TX = encoding_attributes(df_T_ok)
print(TX.shape)
torch.save(TX, os.path.join(output_dir, "heterodata", "features", "TX_only_text_tensor.pt"))
'''


""" EDGES """


def copy_edges(fname_in, fname_out):
    df = pd.read_csv(os.path.join(dir_base, fname_in))
    print(f"{df.shape[0]} edges in {fname_out}")
    df.to_csv(os.path.join(output_dir, fname_out), index=False)


def convert_edges_to_tensors(in_fname, out_fname1, out_fname2):
    df = pd.read_csv(os.path.join(output_dir, "original_data", "edges", in_fname))
    df = df[["src", "tgt"]].copy()
    tensor = edges_encoding(df)
    tensor_rev = edges_rev_encoding(df)
    torch.save(tensor, os.path.join(output_dir, "heterodata", "edgelists", out_fname1))
    torch.save(tensor_rev, os.path.join(output_dir, "heterodata", "edgelists", out_fname2))


"""
copy_edges("tweet_discusses_news_id_politifact.csv", "tweet_discusses_news.csv")
"""
# copy_edges("user_posted_tweet_id_politifact.csv", "user_posted_tweet.csv")
"""
copy_edges("user_retweeted_tweet_id_politifact.csv", "user_retweeted_tweet.csv")
copy_edges("user_mentions_user_id_politifact.csv", "user_mentions_user.csv")
copy_edges("tweet_has_hashtag_id_politifact.csv", "tweet_has_hashtag_hashtag.csv")

convert_edges_to_tensors("tweet_discusses_news.csv", "tweet_discusses_news.pt", "news_is_discussed_by_tweet.pt")
convert_edges_to_tensors("tweet_has_hashtag_hashtag.csv", "tweet_has_hashtag_hashtag.pt","hashtag_is_hashtag_of_tweet.pt")
convert_edges_to_tensors("user_posted_tweet.csv", "user_posted_tweet.pt", "tweet_is_posted_by_user.pt")
convert_edges_to_tensors("user_retweeted_tweet.csv", "user_retweeted_tweet.pt", "tweet_is_retweeted_by_user.pt")
convert_edges_to_tensors("user_mentions_user.csv", "user_mentions_user.pt", "user_is_mentioned_by_user.pt")
"""

""" Applying PCA"""

dim = 256
'''
NX = torch.load(os.path.join(output_dir, "heterodata", "features", "NX_tensor.pt"))
pca_news = PCA(n_components=dim).fit(NX)
torch.save(torch.tensor(pca_news.transform(NX)).float(), os.path.join(output_dir, "heterodata", "features", f"NX_{dim}_tensor.pt"))

TX = torch.load(os.path.join(output_dir, "heterodata", "features", "TX_tensor.pt"))
pca_tweet = PCA(n_components=dim).fit(TX)
torch.save(torch.tensor(pca_tweet.transform(TX)).float(), os.path.join(output_dir, "heterodata", "features", f"TX_{dim}_tensor.pt"))

UX = torch.load(os.path.join(output_dir, "heterodata", "features", "UX_tensor.pt"))
pca_user = PCA(n_components=dim).fit(UX)
torch.save(torch.tensor(pca_user.transform(UX)).float(),
           os.path.join(output_dir, "heterodata", "features", f"UX_{dim}_tensor.pt"))
print("Done")

HX = torch.load(os.path.join(output_dir, "heterodata", "features", "HX_tensor.pt"))
pca_hashtag = PCA(n_components=dim).fit(HX)
torch.save(torch.tensor(pca_hashtag.transform(HX)).float(), os.path.join(output_dir, "heterodata", "features", f"HX_{dim}_tensor.pt"))
'''


def create_mapping():
    df_N = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"))
    mapping_N = df_N['news_id'].to_dict()
    mapping_N_rev = {v: k for k, v in mapping_N.items()}
    save_to_pickle(mapping_N_rev, os.path.join(output_dir, "heterodata", "mappingN.pkl"))

    df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"))
    mapping_U = df_U['user_id'].to_dict()
    mapping_U_rev = {v: k for k, v in mapping_U.items()}
    save_to_pickle(mapping_U_rev, os.path.join(output_dir, "heterodata", "mappingU.pkl"))

    df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"))
    mapping_T = df_T['tweet_id'].to_dict()
    mapping_T_rev = {v: k for k, v in mapping_T.items()}
    save_to_pickle(mapping_T_rev, os.path.join(output_dir, "heterodata", "mappingT.pkl"))

    df_H = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "hashtag.csv"))
    mapping_H = df_H['hashtag_id'].to_dict()
    mapping_H_rev = {v: k for k, v in mapping_H.items()}
    save_to_pickle(mapping_H_rev, os.path.join(output_dir, "heterodata", "mappingH.pkl"))


def mapping_edges(df, mapping_src, mapping_dst):
    df_new = pd.DataFrame(columns=df.columns)
    df_new['src'] = df['src'].map(mapping_src)
    df_new['tgt'] = df['tgt'].map(mapping_dst)
    # df.drop(columns=["source", "target"], axis=1, inplace=True)
    # df['src'] = df['src'].astype(int)
    # df['tgt'] = df['tgt'].astype(int)
    return df_new



def get_sparse_eye(size):
    eye = scipy.sparse.eye(size)
    coo = eye.tocoo()
    values = coo.data
    indices = torch.LongTensor([coo.row, coo.col])
    i = torch.sparse.FloatTensor(indices, torch.FloatTensor(values), torch.Size([size, size]))
    return i


def load_politifact_heterodata(base_dir, mode="EF_256"):

    base_dir = os.path.join(base_dir, 'heterodata')
    nodes_dir = os.path.join(base_dir, 'features')
    edges_dir = os.path.join(base_dir, 'edgelists')

    # Load node features

    dim = 256
    if mode == "EF_all":
        suffix = ""
    elif "meta" in mode:
        suffix = "only_meta_"
    else:
        suffix = f"{dim}_"

    NX = torch.load(os.path.join(nodes_dir, f'NX_{suffix}tensor.pt'))
    NY = torch.load(os.path.join(base_dir, 'NY_tensor.pt'))
    TX = torch.load(os.path.join(nodes_dir, f'TX_{suffix}tensor.pt'))
    UX = torch.load(os.path.join(nodes_dir, f'UX_{suffix}tensor.pt'))
    HX = torch.load(os.path.join(nodes_dir, f'HX_{suffix}tensor.pt'))

    # Load edgelists

    # tweet_discusses_claim
    TN_tensor = torch.load(os.path.join(edges_dir, 'tweet_discusses_news.pt'))
    NT_tensor = torch.load(os.path.join(edges_dir, 'news_is_discussed_by_tweet.pt'))

    # tweet_has_hashtag_hashtag
    TH_tensor = torch.load(os.path.join(edges_dir, 'tweet_has_hashtag_hashtag.pt'))
    HT_tensor = torch.load(os.path.join(edges_dir, 'hashtag_is_hashtag_of_tweet.pt'))

    # user_posted_tweet
    UT_tensor = torch.load(os.path.join(edges_dir, 'user_posted_tweet.pt'))
    TU_tensor = torch.load(os.path.join(edges_dir, 'tweet_is_posted_by_user.pt'))

    # user_retweeted_tweet
    UR_tensor = torch.load(os.path.join(edges_dir, 'user_retweeted_tweet.pt'))
    RU_tensor = torch.load(os.path.join(edges_dir, 'tweet_is_retweeted_by_user.pt'))

    # user_mentions_user
    UU_tensor = torch.load(os.path.join(edges_dir, 'user_mentions_user.pt'))
    UU_rev_tensor = torch.load(os.path.join(edges_dir, 'user_is_mentioned_by_user.pt'))

    data = HeteroData()

    # NODES
    if "only_net" in mode: #unimodal only network
        data["news"].x = get_sparse_eye(NX.shape[0])
        data['news'].y = NY
        data['tweet'].x = get_sparse_eye(TX.shape[0])
        data['user'].x = get_sparse_eye(UX.shape[0])
        data['hashtag'].x = get_sparse_eye(HX.shape[0])

    else: #early fusion
        data['news'].x = NX
        data['news'].y = NY
        data['tweet'].x = TX
        data['user'].x = UX
        data['hashtag'].x = HX


    # EDGES

    data['tweet', 'discusses', 'news'].edge_index = TN_tensor
    data['news', 'is_discussed_by', 'tweet'].edge_index = NT_tensor

    data['tweet', 'has_hashtag', 'hashtag'].edge_index = TH_tensor
    data['hashtag', 'is_hashtag_of', 'tweet'].edge_index = HT_tensor

    data['user', 'posted', 'tweet'].edge_index = UT_tensor
    data['tweet', 'is_posted_by', 'user'].edge_index = TU_tensor

    data['user', 'retweeted', 'tweet'].edge_index = UR_tensor
    data['tweet', 'is_retweeted_by', 'user'].edge_index = RU_tensor


    data['user', 'mentions', 'user'].edge_index = UU_tensor
    data['user', 'is_mentioned_by', 'user'].edge_index = UU_rev_tensor


    # Add metapaths # CTUTC, CTHTC


    metapaths = [
        [('news', 'is_discussed_by', 'tweet'),
         ('tweet', 'is_posted_by', 'user'),
         ('user', 'posted', 'tweet'),
         ('tweet', 'discusses', 'news')], #NTUTN
        [('news', 'is_discussed_by', 'tweet'),
         ('tweet', 'has_hashtag', 'hashtag'),
         ('hashtag', 'is_hashtag_of', 'tweet'),
         ('tweet', 'discusses', 'news')], #NTHTN
        [('news', 'is_discussed_by', 'tweet'),
         ('tweet', 'is_posted_by', 'user'),
         ('user', 'mentions', 'user'),
         ('user', 'posted', 'tweet'),
         ('tweet', 'discusses', 'news')] #NTUUTN
    ]

    if "mps" in mode or "EF" in mode:
        data = AddMetaPaths(metapaths, weighted=True)(data)

    n_samples = len(data["news"])
    transform = T.RandomNodeSplit(num_val=0.10, num_test=0.15) #num_val=n_samples * 0.15, num_test=n_samples * 0.25
    data = transform(data)

    # ensure only multimodal news in the test set
    """
    news_multimodal = pd.read_csv(os.path.join(os.path.dirname(base_dir), 'original_data', 'edges', 'tweet_discusses_news.csv'))['tgt'].drop_duplicates().tolist()
    indices_multimodal = torch.tensor(news_multimodal)
    mask_multimodal = index_to_mask(indices_multimodal)
    old_test_mask = data["news"].test_mask
    data["news"].test_mask = torch.logical_and(mask_multimodal, old_test_mask)
    """
    # ensure no duplicate texts in the test set
    """
    news_all = pd.read_csv(os.path.join(os.path.dirname(base_dir), 'original_data', 'nodes', 'news.csv'))
    indices_no_duplicates = torch.tensor(news_all[~news_all.duplicated('embedding', keep=False)].index.tolist())
    mask_no_duplicates = index_to_mask(indices_no_duplicates)
    old_test_mask = data["news"].test_mask
    data["news"].test_mask = torch.logical_and(mask_no_duplicates, old_test_mask)
    """
    """
    data = data.to(get_device())
    """
    return data

# data = load_politifact_heterodata()
# print(data)

"""
print(torch.min(data['tweet', 'discusses', 'news'].edge_index, dim=1).values)
print(torch.max(data['tweet', 'discusses', 'news'].edge_index, dim=1).values)

print(torch.min(data['user', 'posted', 'tweet'].edge_index, dim=1).values)
print(torch.max(data['user', 'posted', 'tweet'].edge_index, dim=1).values)

print(torch.min(data['user', 'retweeted', 'tweet'].edge_index, dim=1).values)
print(torch.max(data['user', 'retweeted', 'tweet'].edge_index, dim=1).values)

print(torch.min(data['user', 'mentions', 'user'].edge_index, dim=1).values)
print(torch.max(data['user', 'mentions', 'user'].edge_index, dim=1).values)
"""


# create_mapping()


# mappingN = open_pickle(os.path.join(output_dir, "heterodata", "mappingN.pkl"))
# mappingT = open_pickle(os.path.join(output_dir, "heterodata", "mappingT.pkl"))
# mappingU = open_pickle(os.path.join(output_dir, "heterodata", "mappingU.pkl"))
# mappingH = open_pickle(os.path.join(output_dir, "heterodata", "mappingH.pkl"))


def remapping_edges(fname_in, fname_out, src_col, tgt_col, src_mapping, tgt_mapping):
    df = pd.read_csv(os.path.join(dir_base, "file_ausiliari", fname_in))
    df.rename(columns={src_col: "src", tgt_col: "tgt"}, inplace=True)
    df['src'] = df['src'].map(src_mapping)
    df['tgt'] = df['tgt'].map(tgt_mapping)
    df = df.dropna()
    df = df.astype({'src': 'int32', 'tgt': 'int32'})
    # df['src'].replace(src_mapping, inplace=True)
    # df['tgt'].replace(tgt_mapping, inplace=True)
    df.to_csv(os.path.join(output_dir, "original_data", "edges", fname_out), index=False)

# remapping_edges(fname_in="tweet_discusses_news_politifact.csv", fname_out="tweet_discusses_news.csv", src_col="tweet_id", tgt_col="news_id", src_mapping=mappingT, tgt_mapping=mappingN)
# remapping_edges(fname_in="tweet_has_hashtag_politifact.csv", fname_out="tweet_has_hashtag_hashtag.csv", src_col="tweet_id", tgt_col="hashtag_id", src_mapping=mappingT, tgt_mapping=mappingH)
# remapping_edges(fname_in="user_posted_tweet_politifact.csv", fname_out="user_posted_tweet.csv", src_col="user_id", tgt_col="tweet_id", src_mapping=mappingU, tgt_mapping=mappingT)
# remapping_edges(fname_in="user_retweeted_tweet_politifact.csv", fname_out="user_retweeted_tweet.csv", src_col="user_id", tgt_col="tweet_id", src_mapping=mappingU, tgt_mapping=mappingT)
# remapping_edges(fname_in="user_mentions_user_politifact.csv", fname_out="user_mentions_user.csv", src_col="user_id", tgt_col="mentioned_user_id", src_mapping=mappingU, tgt_mapping=mappingU)


def create_masks(seed):
    set_random_seed(seed)
    data = load_politifact_heterodata(output_dir)
    train_mask = mask_to_index(data[target].train_mask) #.tolist()
    val_mask = mask_to_index(data[target].val_mask) #.tolist()
    test_mask = mask_to_index(data[target].test_mask) #.tolist()
    return train_mask, val_mask, test_mask


""" FOR DPSG - v2"""
def create_DPSG_attributes():
    dim_text = 256

    #text_emb_size = 768
    NX = torch.load(os.path.join(output_dir, "heterodata", "features", "NX_only_text_tensor.pt"))
    #print(f"text_emb_size: {NX.shape[1]}")
    #torch.save(NX, os.path.join(output_dir, "heterodata", "features", "DPSG_text_emb_claim.pt"))
    NX_pca = apply_PCA(NX, dim_text)
    print(f"text_emb_size for claim: {NX_pca.shape[1]}")
    torch.save(NX_pca, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_news_{dim_text}.pt"))
    df_N = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"))
    print("Encoding title...")
    df_N = encoding_short_text(df_N, "title", target_dim=256)
    NX2 = encoding_attributes(df_N[['title_encoded']].copy())
    torch.save(NX2, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_news_title_{dim_text}.pt"))


    #image_emb_size = /

    #post_features_emb_size = 6
    """
    df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"))
    df_T.drop(columns = ["text", "tweet_id", "lang", "created_at", "truncated", "text_emb_twhin_bert_base"], axis=1, inplace=True)
    df_T = scale_numeric(df_T, 'retweet_count')
    df_T = scale_numeric(df_T, 'favorite_count')
    TX = encoding_attributes(df_T)
    print(TX.shape)
    torch.save(TX, os.path.join(output_dir, "heterodata", "features", "TX_only_meta_tensor.pt"))
    TX = encoding_attributes(df_T)
    print(f"post_features_emb_size: {TX.shape[1]}")
    torch.save(TX, os.path.join(output_dir, "heterodata", "features", "DPSG_post_features_emb.pt"))
    """
    df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"))
    df_T_ok = df_T[['text_emb_twhin_bert_base']].copy()
    TX = encoding_attributes(df_T_ok)
    TX_pca = apply_PCA(TX, dim_text)
    print(f"text_emb_size for tweet: {TX_pca.shape[1]}")
    torch.save(TX_pca, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_tweet_{dim_text}.pt"))

    #user_features_emb_size = 10
    """
    df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
    df_U = scale_numeric(df_U, 'followers_count')
    df_U = scale_numeric(df_U, 'friends_count')
    df_U = scale_numeric(df_U, 'listed_count')
    df_U = scale_numeric(df_U, 'favourites_count')
    df_U = scale_numeric(df_U, 'statuses_count')
    df_U_ok = df_U.drop(columns=["user_id", "location", "description", "embedding", "created_at"])
    UX = encoding_attributes(df_U_ok)
    print(f"user_feats_emb_size: {UX.shape[1]}")
    torch.save(UX, os.path.join(output_dir, "heterodata", "features", "DPSG_user_features_emb.pt"))
    """
    df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
    print("Encoding location...")
    df_U = encoding_short_text(df_U, "location", target_dim=128)
    print("Encoding description...")
    df_U = encoding_short_text(df_U, "description", target_dim=128)
    df_U_ok = df_U[['location_encoded', 'description_encoded']].copy()
    UX = encoding_attributes(df_U_ok)
    UX_pca = apply_PCA(UX, dim_text)
    print(f"text_emb_size for user: {UX_pca.shape[1]}")
    torch.save(UX_pca, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_user_{dim_text}.pt"))




if __name__ == "__main__":
    dict_split = {}
    for seed in training_seed:
        dict_split[seed] = {}
        dict_split[seed]['train'], dict_split[seed]['val'], dict_split[seed]['test'] = create_masks(seed)
    save_to_pickle(dict_split, os.path.join(output_dir, "heterodata", "masks_indices_75_10_15_all.pkl"))
    #create_DPSG_attributes()




