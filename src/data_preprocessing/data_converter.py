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

dataset = "gossipcop_imgs"
base_dir = f"/mnt/nas/martirano/{dataset}"
input_dir = os.path.join(base_dir, "original_data")
output_dir = os.path.join(base_dir, "heterodata")
target_type = "news"



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


def clean_text(text):
    text = re.sub('RT @[A-Za-z0-9_]+:', '', text)  # retweet
    text = re.sub('@[A-Za-z0-9_]+', '', text)  # tag - users
    text = re.sub('https?://\S+|www\.\S+', '', text)  # url
    text = re.sub('#', '', text)  # hashtags
    # text = emoji.replace_emoji(text, replace='') #emoji
    text = emoji.demojize(text)
    return text


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


def get_sparse_eye(size):
    eye = scipy.sparse.eye(size)
    coo = eye.tocoo()
    values = coo.data
    indices = torch.LongTensor([coo.row, coo.col])
    i = torch.sparse.FloatTensor(indices, torch.FloatTensor(values), torch.Size([size, size]))
    return i


def create_mapping(node_type):
    df = pd.read_csv(os.path.join(input_dir, "nodes", f"{node_type}.csv"))
    mapping = df['id'].to_dict()
    mapping_rev = {v: k for k, v in mapping.items()}
    save_to_pickle(mapping_rev, os.path.join(output_dir, f"mapping{node_type[0].upper()}.pkl"))



def mapping_edges(df, mapping_src, mapping_dst):
    df_new = pd.DataFrame(columns=df.columns)
    df_new['src'] = df['src'].map(mapping_src)
    df_new['tgt'] = df['tgt'].map(mapping_dst)
    df_new = df_new.astype({'src': 'int32', 'tgt': 'int32'})
    return df_new


def edges_encoding(df):
    return torch.tensor(df.values.T)


def edges_rev_encoding(df):
    df_rev = df[['tgt', 'src']]
    return torch.tensor(df_rev.values.T)


def convert_edges_to_tensors(in_fname, out_fname1, out_fname2):
    df = pd.read_csv(os.path.join(input_dir, "edges", in_fname))
    df = df[["src", "tgt"]].copy()
    tensor = edges_encoding(df)
    tensor_rev = edges_rev_encoding(df)
    torch.save(tensor, os.path.join(output_dir, "edgelists", out_fname1))
    torch.save(tensor_rev, os.path.join(output_dir, "edgelists", out_fname2))


def create_masks(seed):
    set_random_seed(seed)
    data = load_heterodata(output_dir)
    train_mask = mask_to_index(data[target_type].train_mask) #.tolist()
    val_mask = mask_to_index(data[target_type].val_mask) #.tolist()
    test_mask = mask_to_index(data[target_type].test_mask) #.tolist()
    return train_mask, val_mask, test_mask


def load_heterodata(heterodata_dir, mode="EF_256"):

    nodes_dir = os.path.join(heterodata_dir, 'features')
    edges_dir = os.path.join(heterodata_dir, 'edgelists')

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


def extract_knowledge(dataset_name, no_snapshot):

    print(f"Processing snapshot {no_snapshot}")
    original_dir = os.path.join(get_base_dir(), dataset_name, 'snapshot_'+str(no_snapshot), 'original_data')
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, 'snapshot_'+str(no_snapshot), 'heterodata')
    heterodata_prev_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{(no_snapshot-1)}', 'heterodata')

    mapping_labels = open_pickle(os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))
    target_type = get_target_type(dataset_name)
    Y_df = pd.read_csv(os.path.join(original_dir, target_type + '_labels.csv'))

    # processing nodes +
    print("processing nodes + mapping...")
    K_new_nodes = {}
    K_old_nodes = {}

    if no_snapshot == 0:
        mapping = {}
        for fname in os.listdir(os.path.join(original_dir, 'nodes')):
            n_type = fname[:-5]  # remove the last 4 characters (".csv")
            f = os.path.join(original_dir, 'nodes', fname)
            df = pd.read_csv(f)
            id_column = next(col for col in df.columns if "id" in col.lower())
            mapping[n_type] = {row[id_column]: idx for idx, row in df.iterrows()}
            K_new_nodes[n_type] = list(mapping[n_type].values())
            K_old_nodes[n_type] = {}

            X = attributes_encoding(df, dataset_name, n_type, no_snapshot)
            torch.save(X, os.path.join(heterodata_dir, 'features', n_type+'s.pt' ))
            #cprint(f'{n_type} features saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
            print(f'{n_type} features saved')

        # mapping labels
        print("mapping labels...")
        #Y['id'] = Y['id'].map(mapping[target_type + 's']) in questo caso non serve
        Y_df['label'] = Y_df['label'].map(mapping_labels)
        Y = torch.tensor(Y_df['label'].values)
        torch.save(Y, os.path.join(heterodata_dir, target_type + '_labels.pt'))
        #cprint(f'Ground truth saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
        print(f'Ground truth saved')

    

    # processing edges
    print("processing edges...")
    K_new_edges = {}
    K_old_edges = {}

    for fname in os.listdir(os.path.join(original_dir, 'edges')):
        e_type = fname[:-4]  # remove the last 4 characters (".csv")
        print(f"processing edge type {e_type}")
        edge_info = extract_edge_info(fname)
        n_type_src = edge_info[0]
        n_type_tgt = edge_info[1]
        f = os.path.join(original_dir, 'edges', fname)
        df = pd.read_csv(f, usecols=['src', 'tgt'])
        print(f"original csv shape: {df.shape}")
        df['src'] = df['src'].map(mapping[n_type_src])
        df['tgt'] = df['tgt'].map(mapping[n_type_tgt])
        K_new_edges[e_type] = df[['src', 'tgt']].values.tolist()
        Xe = edges_encoding(df=df)
        print(f"Dimension of Xe {Xe.shape}")

        if no_snapshot==0:
            K_old_edges[e_type] = []
        else:
            old_old_edges = open_pickle(os.path.join(heterodata_prev_dir, 'K_old_edges.pkl'))[e_type]
            old_new_edges = open_pickle(os.path.join(heterodata_prev_dir, 'K_new_edges.pkl'))[e_type]
            K_old_edges[e_type] = old_old_edges + old_new_edges
            Xe_prev = torch.load(os.path.join(heterodata_prev_dir, 'edgelists', e_type+'.pt'))
            print(f"Dimension of Xe_prev {Xe_prev.shape}")
            Xe = torch.cat((Xe_prev, Xe), dim=1)

        torch.save(Xe, os.path.join(heterodata_dir, 'edgelists', e_type + '.pt'))
        #cprint(f'{e_type} edgelist saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
        print(f'{e_type} edgelist saved')

    # saving K_new_edges, K_old_edges
    save_dict_to_pickle(K_new_edges, os.path.join(heterodata_dir, 'K_new_edges.pkl'))
    save_dict_to_pickle(K_old_edges, os.path.join(heterodata_dir, 'K_old_edges.pkl'))



if __name__ == "__main__":

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

    # processing only_meta
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
    # processing only_text
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
    # processing timeline
    '''
    df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), dtype={'embedding': str})
    df_U_ok = df_U[['embedding']].copy()
    UX = encoding_attributes(df_U_ok)
    print(UX.shape)
    torch.save(UX, os.path.join(output_dir, "heterodata", "features", "UX_only_timeline_tensor.pt"))
    '''

    """ Applying PCA"""

    dim = 256
    '''
    HX = torch.load(os.path.join(output_dir, "heterodata", "features", "HX_tensor.pt"))
    pca_hashtag = PCA(n_components=dim).fit(HX)
    torch.save(torch.tensor(pca_hashtag.transform(HX)).float(), os.path.join(output_dir, "heterodata", "features", f"HX_{dim}_tensor.pt"))
    '''

    # create_mapping()

    # mappingN = open_pickle(os.path.join(output_dir, "heterodata", "mappingN.pkl"))
    # mappingT = open_pickle(os.path.join(output_dir, "heterodata", "mappingT.pkl"))
    # mappingU = open_pickle(os.path.join(output_dir, "heterodata", "mappingU.pkl"))
    # mappingH = open_pickle(os.path.join(output_dir, "heterodata", "mappingH.pkl"))

    # remapping_edges(fname_in="tweet_discusses_news_politifact.csv", fname_out="tweet_discusses_news.csv", src_col="tweet_id", tgt_col="news_id", src_mapping=mappingT, tgt_mapping=mappingN)
    # remapping_edges(fname_in="tweet_has_hashtag_politifact.csv", fname_out="tweet_has_hashtag_hashtag.csv", src_col="tweet_id", tgt_col="hashtag_id", src_mapping=mappingT, tgt_mapping=mappingH)
    # remapping_edges(fname_in="user_posted_tweet_politifact.csv", fname_out="user_posted_tweet.csv", src_col="user_id", tgt_col="tweet_id", src_mapping=mappingU, tgt_mapping=mappingT)
    # remapping_edges(fname_in="user_retweeted_tweet_politifact.csv", fname_out="user_retweeted_tweet.csv", src_col="user_id", tgt_col="tweet_id", src_mapping=mappingU, tgt_mapping=mappingT)
    # remapping_edges(fname_in="user_mentions_user_politifact.csv", fname_out="user_mentions_user.csv", src_col="user_id", tgt_col="mentioned_user_id", src_mapping=mappingU, tgt_mapping=mappingU)

    dict_split = {}
    for seed in training_seed:
        dict_split[seed] = {}
        dict_split[seed]['train'], dict_split[seed]['val'], dict_split[seed]['test'] = create_masks(seed)
    save_to_pickle(dict_split, os.path.join(output_dir, "heterodata", "masks_indices_75_10_15_all.pkl"))
    #create_DPSG_attributes()




