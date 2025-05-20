import zipfile
from io import BytesIO
import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch_geometric.transforms as T
from torch_geometric.utils import mask_to_index

from src.data_preprocessing.utils import scale_numeric, apply_PCA
from src.utils import set_random_seed, training_seed, save_to_pickle

input_dir = '/mnt/nas/pisani/dataset_mumin/mumin-small.v0.zip'
output_dir = '/mnt/nas/martirano/mumin/heterodata' #'/mnt/nas/martirano/mumin'
target = "claim"


def read_pickle_from_zip_folder(fname):
    # Open the zip file
    with zipfile.ZipFile(input_dir, 'r') as zip_ref:
        with zip_ref.open(fname) as file:
            # Read the file content into a pandas dataframe
            return pd.read_pickle(BytesIO(file.read()))

# conversion from datetime64[ns] to timestamp + normalization
# boost: replace nan values with the mean of the column instead of epoch start to avoid otliers (MinMax Scaling is sensitive to outliers)
def convert_datetime_to_timestamp(df, col):
    #df[col+'_timestamp'] = df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else pd.Timestamp('1970-01-01 00:00:00').timestamp())
    mean_date = df[col].mean()
    df[col] = df[col].fillna(mean_date)
    df[col+'_timestamp'] = df[col].apply(lambda x: x.timestamp())
    scaler = MinMaxScaler()
    df[col+'_timestamp_scaled'] = scaler.fit_transform(df[[col+'_timestamp']])
    df.drop([col, col+'_timestamp'], axis=1, inplace=True)
    return df

def parsing_embedding_float(df, col):
    df[col+'_parsed'] = df.apply(lambda x: torch.tensor(x[col], dtype=torch.float32), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df

def parsing_embedding_int(df, col):
    df[col + '_parsed'] = df.apply(lambda x: torch.tensor(x[col], dtype=torch.int32), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df


def one_hot_encoding(df, col):
    df[col] = df[col].astype(str)  # added for int values (cluster_labels)
    if df[col].dtype.name == 'category':
        unique_categories = df[col].cat.categories.tolist()
    else:
        unique_categories = df[col].unique().tolist()
    num_categories = len(unique_categories)
    category_to_index = {category: idx for idx, category in enumerate(unique_categories)}

    def one_hot_encode(category):
        category = str(category)  # Ensure the category is treated as a string
        one_hot_vector = np.zeros(num_categories, dtype=int)
        one_hot_vector[category_to_index[category]] = 1
        # return one_hot_vector
        return torch.tensor(one_hot_vector, dtype=torch.int32)

    df[col + '_ohe'] = df.apply(lambda x: one_hot_encode(x[col]), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df


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
        elif df[col].dtype == 'object':
            embedding_tensors = df[col].apply(lambda x: torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x)
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



""" FOR DPSG - v2"""
def create_DPSG_attributes():

    # nodes
    df_C = read_pickle_from_zip_folder('claim.pickle')
    df_T = read_pickle_from_zip_folder('tweet.pickle')
    df_U = read_pickle_from_zip_folder('user.pickle')
    #df_I = read_pickle_from_zip_folder('image.pickle')

    dim_text = 256


    """ FOR DPSG - v2"""

    #text_emb_size = 768
    CX = torch.load(os.path.join(output_dir, "heterodata", "features", "CX_only_text_tensor.pt"))
    CX_pca = apply_PCA(CX, dim_text)
    print(f"text_emb_size for claim: {CX_pca.shape[1]}")
    torch.save(CX_pca, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_claim_{dim_text}.pt"))

    #image_emb_size = /

    #post_features_emb_size = 3
    """
    df_T_sub = df_T.drop(columns=['tweet_id', 'text', 'text_emb', 'lang', 'lang_emb', 'created_at', 'source'], axis=1)
    df_T_sub = scale_numeric(df_T_sub, 'num_retweets') #POST-FEATURES
    df_T_sub = scale_numeric(df_T_sub, 'num_replies') #POST-FEATURES
    df_T_sub = scale_numeric(df_T_sub, 'num_quote_tweets') #POST-FEATURES
    TX = encoding_attributes(df_T_sub)
    print(f"post_features_emb_size: {TX.shape[1]}")
    torch.save(TX, os.path.join(output_dir, "heterodata", "features", "DPSG_post_features_emb.pt"))
    """
    TX = torch.load(os.path.join(output_dir, "heterodata", "features", "TX_only_text_tensor.pt"))
    TX_pca = apply_PCA(TX, dim_text)
    print(f"text_emb_size for tweet: {TX_pca.shape[1]}")
    torch.save(TX_pca, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_tweet_{dim_text}.pt"))

    """
    #user_features_emb_size = 6
    df_U_sub = df_U.drop(columns=['user_id', 'username', 'description', 'url',	'name', 'description_emb', 'location', 'created_at'], axis=1)
    df_U_sub = scale_numeric(df_U_sub, 'num_followers') #USER-FEATURES
    df_U_sub = scale_numeric(df_U_sub, 'num_followees') #USER-FEATURES
    df_U_sub = scale_numeric(df_U_sub, 'num_tweets') #USER-FEATURES
    df_U_sub = scale_numeric(df_U_sub, 'num_listed') #USER-FEATURES
    UX = encoding_attributes(df_U_sub)
    print(f"user_feats_emb_size: {UX.shape[1]}")
    torch.save(UX, os.path.join(output_dir, "heterodata", "features", "DPSG_user_features_emb.pt"))
    """
    UX = torch.load(os.path.join(output_dir, "heterodata", "features", "UX_only_text_tensor.pt"))
    UX_pca = apply_PCA(UX, dim_text)
    print(f"text_emb_size for user: {UX_pca.shape[1]}")
    torch.save(UX_pca, os.path.join(output_dir, "heterodata", "features", f"DPSG_text_emb_user_{dim_text}.pt"))


    """ CLAIM """
    """
    #processing only_meta
    print('##### preprocessing claim fetaures only metadata... #####')
    print(df_C.columns)
    df_C_sub = df_C.drop(columns=['embedding', 'label', 'reviewer_emb', 'train_mask', 'val_mask', 'test_mask'], axis=1)
    df_C_sub = encoding_short_text(df_C_sub, 'reviewers') #114 distinct values
    df_C_sub = convert_datetime_to_timestamp(df_C_sub, 'date')
    df_C_sub = one_hot_encoding(df_C_sub,'language') #35 distinct values
    df_C_sub = encoding_short_text(df_C_sub, 'keywords')
    df_C_sub = encoding_short_text(df_C_sub, 'cluster_keywords', target_dim=64) #26 distinct values, but multiple strings each
    CX = encoding_attributes(df_C_sub)
    print('claim only metadata:', CX.shape)
    torch.save(CX, os.path.join(output_dir, "features", "CX_only_meta_tensor.pt"))
    
    #processing only_text
    print('##### preprocessing claim fetaures only text... #####')
    df_C_sub = parsing_embedding_float(df_C, 'embedding')
    CX = encoding_attributes(df_C_sub[['embedding_parsed']].copy())
    print('claim only text:', CX.shape)
    torch.save(CX, os.path.join(output_dir, "features", "CX_only_text_tensor.pt")) #TEXT??? Serve PCA?
    """

    """ TWEET """
    """
    #processing only_meta
    print('##### preprocessing tweet fetaures only metadata... #####')
    df_T_sub = df_T.drop(columns=['tweet_id', 'text', 'text_emb', 'lang'], axis=1)
    df_T_sub['lang_emb'] = df_T_sub.apply(lambda x: torch.tensor(x['lang_emb'], dtype=torch.int32), axis=1)
    df_T_sub = convert_datetime_to_timestamp(df_T_sub, 'created_at')
    df_T_sub = encoding_short_text(df_T_sub, 'source') #136 distinct values
    df_T_sub = scale_numeric(df_T_sub, 'num_retweets') #POST-FEATURES
    df_T_sub = scale_numeric(df_T_sub, 'num_replies') #POST-FEATURES
    df_T_sub = scale_numeric(df_T_sub, 'num_quote_tweets') #POST-FEATURES
    TX = encoding_attributes(df_T_sub)
    print('tweet only metadata:', TX.shape)
    torch.save(TX, os.path.join(output_dir, "features", "TX_only_meta_tensor.pt"))
    
    #processing only_text
    print('##### preprocessing tweet fetaures... #####')
    df_T_sub = parsing_embedding_float(df_T, 'text_emb')
    TX = encoding_attributes(df_T_sub[['text_emb_parsed']].copy())
    print('tweet only text:', TX.shape)
    torch.save(TX, os.path.join(output_dir, "features", "TX_only_text_tensor.pt"))
    """

    """ USER """
    """
    #processing only_meta
    print('##### preprocessing user fetaures only metadata... #####')
    df_U_sub = df_U.drop(columns=['user_id', 'username', 'description', 'url',	'name', 'description_emb', 'location'], axis=1)
    df_U_sub = convert_datetime_to_timestamp(df_U_sub, 'created_at')
    df_U_sub = scale_numeric(df_U_sub, 'num_followers') #USER-FEATURES
    df_U_sub = scale_numeric(df_U_sub, 'num_followees') #USER-FEATURES
    df_U_sub = scale_numeric(df_U_sub, 'num_tweets') #USER-FEATURES
    df_U_sub = scale_numeric(df_U_sub, 'num_listed') #USER-FEATURES
    UX = encoding_attributes(df_U_sub)
    print('user only metadata:', UX.shape)
    torch.save(UX, os.path.join(output_dir, "features", "UX_only_meta_tensor.pt"))
    
    #processing only_text
    print('##### preprocessing user fetaures... #####')
    df_U_sub = encoding_short_text(df_U, 'location')
    df_U_sub['description_emb'] = df_U_sub.apply(lambda x: torch.tensor(x['description_emb'], dtype=torch.float32), axis=1)
    UX = encoding_attributes(df_U_sub[['location_encoded', 'description_emb']].copy())
    print('user only text:', UX.shape)
    torch.save(UX, os.path.join(output_dir, "features", "UX_only_text_tensor.pt"))
    """


    """ IMAGE """
    """
    print('##### preprocessing image fetaures... #####')
    df_I_sub = df_I.drop(columns=['url', 'pixels'], axis=1)
    df_I_sub = scale_numeric(df_I_sub, 'width')
    df_I_sub = scale_numeric(df_I_sub, 'height')
    df_I_sub['pixels_emb'] = df_I_sub.apply(lambda x: torch.tensor(x['pixels_emb'], dtype=torch.float32), axis=1)
    #df_I_sub.to_csv('../data/features/IX.csv', index=False)
    IX = encoding_attributes(df_I_sub)
    torch.save(IX, '../data/features/IX_tensor.pt')
    print('hashtag:', IX.shape)
    print()
    """
    # Improve image features
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    import numpy as np
    import pandas as pd
    
    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
    model.eval()
    
    # Transformations for the images
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def extract_embedding(image):
        image = preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = model(image)
        return embedding.squeeze().numpy()
    
    # the 'pixels' column contains image arrays
    df_I = read_pickle_from_zip_folder('image.pickle')
    df_I['pixels_encoded'] = df_I['pixels'].apply(lambda x: extract_embedding(np.array(x)))
    
    df_I_sub = df_I.drop(columns=['url', 'pixels', 'pixels_emb'], axis=1)
    df_I_sub = scale_numeric(df_I_sub, 'width')
    df_I_sub = scale_numeric(df_I_sub, 'height')
    
    IX = encoding_attributes(df_I_sub)
    torch.save(IX, '../data/features/IX_tensor_v2.pt')
    print('hashtag:', IX.shape)
    """


def load_mumin_heterodata(base_dir):
    nodes_dir = os.path.join(base_dir, 'features')
    edges_dir = os.path.join(base_dir, 'edgelists')

    # Load node features

    dim = 256

    CX = torch.load(os.path.join(nodes_dir, 'CX_'+str(dim)+'_tensor.pt'))
    CY = torch.load(os.path.join(nodes_dir, 'CY_tensor.pt'))
    TX = torch.load(os.path.join(nodes_dir, 'TX_' + str(dim) + '_tensor.pt'))
    RX = torch.load(os.path.join(nodes_dir, 'RX_'+str(dim)+'_tensor.pt'))
    UX = torch.load(os.path.join(nodes_dir, 'UX_'+str(dim)+'_tensor.pt'))
    HX = torch.load(os.path.join(nodes_dir, 'HX_'+str(dim)+'_tensor_v2.pt'))
    AX = torch.load(os.path.join(nodes_dir, 'AX_'+str(dim)+'_tensor.pt'))
    IX = torch.load(os.path.join(nodes_dir, 'IX_'+str(dim)+'_tensor.pt'))

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

    data = AddMetaPaths(metapaths, weighted=True)(data)

    #n_samples = len(data["claim"])
    transform = T.RandomNodeSplit(num_val=0.10, num_test=0.15) #num_val=n_samples * 0.15, num_test=n_samples * 0.25
    data = transform(data)

    return data


def create_masks(seed):
    set_random_seed(seed)
    data = load_mumin_heterodata(output_dir)
    train_mask = mask_to_index(data[target].train_mask) #.tolist()
    val_mask = mask_to_index(data[target].val_mask) #.tolist()
    test_mask = mask_to_index(data[target].test_mask) #.tolist()
    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    dict_split = {}
    for seed in training_seed:
        dict_split[seed] = {}
        dict_split[seed]['train'], dict_split[seed]['val'], dict_split[seed]['test'] = create_masks(seed)
    save_to_pickle(dict_split, os.path.join(output_dir, "masks_indices_75_10_15.pkl"))
    # create_DPSG_attributes()