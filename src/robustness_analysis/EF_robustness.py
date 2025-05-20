import torch
from torch_geometric.utils import mask_to_index, index_to_mask


"""drop target nodes from training set"""
""" drop_percentage in [0-100]"""
def masking_multimodal(data, target_type, drop_percentage):
    if drop_percentage != 0:
        train_mask_old = data[target_type].train_mask
        train_indices = mask_to_index(train_mask_old)
        num_indices_to_keep = int(len(train_indices) * (1-drop_percentage) / 100)
        selected_train_indices = train_indices[torch.randperm(len(train_indices))[:num_indices_to_keep]]
        train_mask_new = torch.zeros_like(train_mask_old, dtype=torch.bool)
        train_mask_new[selected_train_indices] = True
        data[target_type].train_mask = train_mask_new
    return data

"""half masked unimodal text and half masked unimodal network"""
""" drop_percentage in [0-100]"""
def masking_multimodal_mixed(data, target_type, drop_percentage):
    if drop_percentage != 0:
        data = masking_unimodal_text(data, target_type, drop_percentage/2)
        data = masking_unimodal_network(data, target_type, drop_percentage/2, not_masked=False)
    return data


"""mask nodes features from training set"""
""" drop_percentage in [0-100]"""
def masking_unimodal_text(data, target_type, drop_percentage):
    if drop_percentage != 0:
        train_mask_old = data[target_type].train_mask
        train_indices = mask_to_index(train_mask_old)
        num_indices_to_mask = int(len(train_indices) * drop_percentage / 100)
        selected_masked_indices = train_indices[torch.randperm(len(train_indices))[:num_indices_to_mask]]
        x = data[target_type].x
        features_dim = x.shape[1]
        data[target_type].x[selected_masked_indices] = torch.zeros(num_indices_to_mask, features_dim, dtype=x.dtype, device=x.device)
    return data


"""isolate nodes from training set"""
""" drop_percentage in [0-100]"""
def masking_unimodal_network(data, target_type, drop_percentage, not_masked = True):
    if drop_percentage != 0:
        train_mask_old = data[target_type].train_mask
        train_indices = mask_to_index(train_mask_old)
        num_indices_to_isolate= int(len(train_indices) * drop_percentage / 100)
        if not_masked:
            selected_isolated_indices = train_indices[torch.randperm(len(train_indices))[:num_indices_to_isolate]]
        else:
            non_zero_mask = (data[target_type].x != 0).all(dim=1)
            non_zero_indices = train_indices[non_zero_mask[train_indices]] #indices where features are NOT zeros
            selected_isolated_indices = non_zero_indices[torch.randperm(len(non_zero_indices))[:num_indices_to_isolate]]
        for edge_type in data.edge_types:
            src_type, rel, dst_type = edge_type
            if src_type == target_type or dst_type == target_type:
                edge_index = data[edge_type].edge_index  # Get current edges
                mask = ~(
                        torch.isin(edge_index[0], selected_isolated_indices) |
                        torch.isin(edge_index[1], selected_isolated_indices)
                ) #Keep only edges where neither node is in selected_isolated_nodes
                data[edge_type].edge_index = edge_index[:, mask]
                if 'edge_weight' in data[edge_type]: #for meta-paths
                    data[edge_type].edge_weight = data[edge_type].edge_weight[mask]
    return data



if __name__ == "__main__":

    #TESTING UNIMODAL NETWORK
    """
    data = load_politifact_heterodata(get_base_dir())
    print(data)
    tgt_type = "news"
    data = masking_unimodal_network(data, tgt_type, 10, not_masked = True)
    print(data)
    """

    #TESTING UNIMODAL TEXT
    """
    data = load_politifact_heterodata(get_base_dir())
    tgt_type = "news"
    training_set = data[tgt_type].train_mask
    dim_train_set = torch.sum(training_set).item()
    print(f"Dimensione del training set:{dim_train_set}")
    feats = data[tgt_type].x
    feats0 = torch.all(feats==0, dim=1)
    num_feats0 = torch.sum(feats0).item()
    print(f"Numero di features a 0 del training set prima:{num_feats0}")
    data = masking_unimodal_text(data, tgt_type, 10)
    feats = data[tgt_type].x
    feats0 = torch.all(feats==0, dim=1)
    num_feats0 = torch.sum(feats0).item()
    print(f"Numero di features a 0 del training set dopo:{num_feats0}")
    """

    #TESTING MULTIMODAL
    """
    data = load_politifact_heterodata(get_base_dir())
    tgt_type = "news"
    training_set = data[tgt_type].train_mask
    dim_train_set = torch.sum(training_set).item()
    print(f"Dimensione del training set prima:{dim_train_set}")
    data = masking_multimodal(data, tgt_type, 10)
    training_set = data[tgt_type].train_mask
    dim_train_set = torch.sum(training_set).item()
    print(f"Dimensione del training set dopo:{dim_train_set}")
    """

    #TESTING MULTIMODAL v2
    """
    data = load_politifact_heterodata(get_base_dir())
    print(data)
    tgt_type = "news"
    training_set = data[tgt_type].train_mask
    dim_train_set = torch.sum(training_set).item()
    print(f"Dimensione del training set prima:{dim_train_set}")
    feats = data[tgt_type].x
    feats0 = torch.all(feats == 0, dim=1)
    num_feats0 = torch.sum(feats0).item()
    print(f"Numero di features a 0 del training set prima:{num_feats0}")
    data = masking_multimodal_mixed(data, tgt_type, 10)
    print(data)
    training_set = data[tgt_type].train_mask
    dim_train_set = torch.sum(training_set).item()
    print(f"Dimensione del training set dopo:{dim_train_set}")
    feats = data[tgt_type].x
    feats0 = torch.all(feats == 0, dim=1)
    num_feats0 = torch.sum(feats0).item()
    print(f"Numero di features a 0 del training set dopo:{num_feats0}")
    """
