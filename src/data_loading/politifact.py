import os
import torch
import scipy
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask

from src.utils import get_device, get_base_dir, load_from_pickle, save_to_pickle


def load_mask(fname, type, seed):
    df = pd.read_csv(fname)
    return torch.tensor(df[f"{type}_mask_LP_{seed}"].values, dtype=torch.bool)


def get_sparse_eye(size):
    eye = scipy.sparse.eye(size)
    coo = eye.tocoo()
    values = coo.data
    indices = torch.LongTensor([coo.row, coo.col])
    i = torch.sparse.FloatTensor(indices, torch.FloatTensor(values), torch.Size([size, size]))
    return i



#modes = ["only_net_edges", "only_net_edges_mps", "only_net_edges_meta", "only_net_edges_mps_meta", "EF_256", "EF_all"]
def load_politifact_heterodata(base_dir, split="60_15_25", mode="EF_256", seed=42):
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

    """metapaths = [
                 [('user', 'retweeted', 'tweet'),
                 ('tweet', 'is_posted_by', 'user')]
                ]"""

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

    #if "mps" in mode or "EF" in mode:
    #    data = AddMetaPaths(metapaths, weighted=True)(data)

    """
    n_samples = len(data["news"])
    transform = T.RandomNodeSplit(num_val=0.15, num_test=0.25) #num_val=0.10, num_test=0.15
    data = transform(data)
    """
    indices_masks = load_from_pickle(os.path.join(base_dir, f'masks_indices_{split}.pkl'))
    mask_dim = data["news"].x.shape[0]
    data['news'].train_mask = index_to_mask(indices_masks[seed]["train"], mask_dim)
    data['news'].val_mask = index_to_mask(indices_masks[seed]["val"], mask_dim)
    data['news'].test_mask = index_to_mask(indices_masks[seed]["test"], mask_dim)

    """ ALREADY INCLUDED
    # ensure only multimodal news in the test set
    news_multimodal = pd.read_csv(os.path.join(os.path.dirname(base_dir), 'original_data', 'edges', 'tweet_discusses_news.csv'))['tgt'].drop_duplicates().tolist()
    indices_multimodal = torch.tensor(news_multimodal)
    mask_multimodal = index_to_mask(indices_multimodal, mask_dim)
    old_test_mask = index_to_mask(data["news"].test_mask, mask_dim)
    data["news"].test_mask = torch.logical_and(mask_multimodal, old_test_mask)

    # ensure no duplicate texts in the test set
    news_all = pd.read_csv(os.path.join(os.path.dirname(base_dir), 'original_data', 'nodes', 'news.csv'))
    indices_no_duplicates = torch.tensor(news_all[~news_all.duplicated('embedding', keep=False)].index.tolist())
    mask_no_duplicates = index_to_mask(indices_no_duplicates, mask_dim)
    old_test_mask = data["news"].test_mask
    data["news"].test_mask = torch.logical_and(mask_no_duplicates, old_test_mask)
    """

    data = data.to(get_device())

    return data




if __name__ == "__main__":

    data = load_politifact_heterodata(get_base_dir())
    print(data)