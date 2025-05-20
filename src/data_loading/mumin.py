import os
import torch
import scipy
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch_geometric.transforms as T

from torch_geometric.utils import index_to_mask

from src.utils import load_from_pickle, get_device, get_base_dir


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
def load_mumin_heterodata(base_dir, split="60_15_25", mode="EF_256", seed=42):
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

    CX = torch.load(os.path.join(nodes_dir, f'CX_{suffix}tensor.pt'))
    CY = torch.load(os.path.join(nodes_dir, f'CY_tensor.pt'))
    TX = torch.load(os.path.join(nodes_dir, f'TX_{suffix}tensor.pt'))
    RX = torch.load(os.path.join(nodes_dir, f'RX_{suffix}tensor.pt'))
    UX = torch.load(os.path.join(nodes_dir, f'UX_{suffix}tensor.pt'))
    HX = torch.load(os.path.join(nodes_dir, f'HX_{suffix}tensor_v2.pt'))
    AX = torch.load(os.path.join(nodes_dir, f'AX_{suffix}tensor.pt'))
    IX = torch.load(os.path.join(nodes_dir, f'IX_{suffix}tensor.pt'))

    # Load edgelists

    # tweet_discusses_claim
    TC_tensor = torch.load(os.path.join(edges_dir, 'TC_tensor.pt'))
    CT_tensor = torch.load(os.path.join(edges_dir, 'CT_tensor.pt'))

    # reply_reply_to_tweet
    TR_r_tensor = torch.load(os.path.join(edges_dir, 'TR_r_tensor.pt'))
    RT_r_tensor = torch.load(os.path.join(edges_dir, 'RT_r_tensor.pt'))

    # reply_quote_of_tweet
    TR_q_tensor = torch.load(os.path.join(edges_dir, 'TR_q_tensor.pt'))
    RT_q_tensor = torch.load(os.path.join(edges_dir, 'RT_q_tensor.pt'))

    # tweet_has_hashtag_hashtag
    TH_tensor = torch.load(os.path.join(edges_dir, 'TH_tensor.pt'))
    HT_tensor = torch.load(os.path.join(edges_dir, 'HT_tensor.pt'))

    # tweet_has_article_article
    TA_tensor = torch.load(os.path.join(edges_dir, 'TA_tensor.pt'))
    AT_tensor = torch.load(os.path.join(edges_dir, 'AT_tensor.pt'))

    # tweet_has_image_image
    TI_tensor = torch.load(os.path.join(edges_dir, 'TI_tensor.pt'))
    IT_tensor = torch.load(os.path.join(edges_dir, 'IT_tensor.pt'))

    # tweet_mentions_user
    TU_m_tensor =torch.load(os.path.join(edges_dir, 'TU_m_tensor.pt'))
    UT_m_tensor =torch.load(os.path.join(edges_dir, 'UT_m_tensor.pt'))

    # user_posted_tweet
    UT_tensor =torch.load(os.path.join(edges_dir, 'UT_tensor.pt'))
    TU_tensor =torch.load(os.path.join(edges_dir, 'TU_tensor.pt'))

    # user_posted_reply
    UR_tensor =torch.load(os.path.join(edges_dir, 'UR_tensor.pt'))
    RU_tensor =torch.load(os.path.join(edges_dir, 'RU_tensor.pt'))

    # user_retweeted_tweet
    UT_r_tensor =torch.load(os.path.join(edges_dir, 'UT_r_tensor.pt'))
    TU_r_tensor =torch.load(os.path.join(edges_dir, 'TU_r_tensor.pt'))

    # user_follows_user
    UU_f_tensor =torch.load(os.path.join(edges_dir, 'UU_f_tensor.pt'))
    UU_f_rev_tensor =torch.load(os.path.join(edges_dir, 'UU_f_rev_tensor.pt'))

    # user_mentions_user
    UU_m_tensor =torch.load(os.path.join(edges_dir, 'UU_m_tensor.pt'))
    UU_m_rev_tensor =torch.load(os.path.join(edges_dir, 'UU_m_rev_tensor.pt'))

    # user_has_hashtag_hashtag
    UH_tensor =torch.load(os.path.join(edges_dir, 'UH_tensor.pt'))
    HU_tensor =torch.load(os.path.join(edges_dir, 'HU_tensor.pt'))

    data = HeteroData()

    # NODES

    if "only_net" in mode: #unimodal only network
        data["claim"].x = get_sparse_eye(CX.shape[0])
        data['claim'].y = CY
        data['tweet'].x = get_sparse_eye(TX.shape[0])
        data['reply'].x = get_sparse_eye(RX.shape[0])
        data['user'].x = get_sparse_eye(UX.shape[0])
        data['hashtag'].x = get_sparse_eye(HX.shape[0])
        data['article'].x = get_sparse_eye(AX.shape[0])
        data['image'].x = get_sparse_eye(IX.shape[0])

    else: #early fusion
        data['claim'].x = CX  # torch.tensor(pca_claim.transform(CX)).float()
        data['claim'].y = CY
        data['tweet'].x = TX  # torch.tensor(pca_tweet.transform(TX)).float()
        data['reply'].x = RX  # torch.tensor(pca_reply.transform(RX)).float()
        data['user'].x = UX  # torch.tensor(pca_user.transform(UX)).float()
        data['hashtag'].x = HX  # torch.tensor(pca_hashtag.transform(HX)).float()
        data['article'].x = AX  # torch.tensor(pca_article.transform(AX)).float()
        data['image'].x = IX  # torch.tensor(pca_image.transform(IX)).float()

    # EDGES

    data['tweet', 'discusses', 'claim'].edge_index = TC_tensor
    data['claim', 'is_discussed_by', 'tweet'].edge_index = CT_tensor

    data['reply', 'reply_to', 'tweet'].edge_index = TR_r_tensor
    data['tweet', 'is_replied_by', 'reply'].edge_index = RT_r_tensor

    data['reply', 'quote_of', 'tweet'].edge_index = TR_q_tensor
    data['tweet', 'is_quoted_by', 'reply'].edge_index = RT_q_tensor

    data['tweet', 'has_hashtag', 'hashtag'].edge_index = TH_tensor
    data['hashtag', 'is_hashtag_of', 'tweet'].edge_index = HT_tensor

    data['tweet', 'has_article', 'article'].edge_index = TA_tensor
    data['article', 'is_article_of', 'tweet'].edge_index = AT_tensor

    data['tweet', 'has_image', 'image'].edge_index = TI_tensor
    data['image', 'is_image_of', 'tweet'].edge_index = IT_tensor

    data['tweet', 'mentions', 'user'].edge_index = TU_m_tensor
    data['user', 'is_mentioned_by', 'tweet'].edge_index = UT_m_tensor

    data['user', 'posted', 'tweet'].edge_index = UT_tensor
    data['tweet', 'is_posted_by', 'user'].edge_index = TU_tensor

    data['user', 'posted', 'reply'].edge_index = UR_tensor
    data['reply', 'is_posted_by', 'user'].edge_index = RU_tensor

    data['user', 'retweeted', 'tweet'].edge_index = UT_r_tensor
    data['tweet', 'is_retweeted_by', 'user'].edge_index = TU_r_tensor

    data['user', 'follows', 'user'].edge_index = UU_f_tensor
    data['user', 'is_followed_by', 'user'].edge_index = UU_f_rev_tensor

    data['user', 'mentions', 'user'].edge_index = UU_m_tensor
    data['user', 'is_mentioned_by', 'user'].edge_index = UU_m_rev_tensor

    data['user', 'has_hashtag', 'hashtag'].edge_index = UH_tensor
    data['hashtag', 'is_hashtag_of', 'user'].edge_index = HU_tensor

    # Add metapaths # CTUTC, CTHTC

    metapaths = [[('claim', 'is_discussed_by', 'tweet'),
                  ('tweet', 'is_posted_by', 'user'),
                  ('user', 'posted', 'tweet'),
                  ('tweet', 'discusses', 'claim')],  # CTUTC
                 [('claim', 'is_discussed_by', 'tweet'),
                  ('tweet', 'has_hashtag', 'hashtag'),
                  ('hashtag', 'is_hashtag_of', 'tweet'),
                  ('tweet', 'discusses', 'claim')],  # CTHTC
                 [('claim', 'is_discussed_by', 'tweet'),
                  ('tweet', 'is_replied_by', 'reply'),
                  ('reply', 'reply_to', 'tweet'),
                  ('tweet', 'discusses', 'claim')],  # CTRTC_r
                 [('claim', 'is_discussed_by', 'tweet'),
                  ('tweet', 'is_quoted_by', 'reply'),
                  ('reply', 'quote_of', 'tweet'),
                  ('tweet', 'discusses', 'claim')]]  # CTRTC_q

    #if "mps" in mode or "EF" in mode:
        #data = AddMetaPaths(metapaths, weighted=True)(data)

    """
    n_samples = len(data["claim"])
    transform = T.RandomNodeSplit(num_val=0.10, num_test=0.15) #num_val=0.15, num_test=0.25
    data = transform(data)
    """

    indices_masks = load_from_pickle(os.path.join(base_dir, f'masks_indices_{split}.pkl'))
    mask_dim = data["claim"].x.shape[0]
    data['claim'].train_mask = index_to_mask(indices_masks[seed]["train"], mask_dim)
    data['claim'].val_mask = index_to_mask(indices_masks[seed]["val"], mask_dim)
    data['claim'].test_mask = index_to_mask(indices_masks[seed]["test"], mask_dim)


    """ QUESTO OK
    # ensure only multimodal news in the test set
    news_multimodal = pd.read_csv(os.path.join(os.path.dirname(base_dir), 'original_data', 'edges', 'tweet_discusses_news.csv'))[
        'tgt'].drop_duplicates().tolist()
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
    data = load_mumin_heterodata(get_base_dir())
    print(data)

