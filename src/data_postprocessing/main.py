import os.path

import torch
from torch_geometric.nn import to_hetero
from src.models.GAT import GAT

from src.data_postprocessing.GNN_explanation import get_explanation, compute_cumulative_contributions, \
    compute_normalized_node_vs_edge_importance, plot_nodes_vs_edges_importance, plot_types_importance
from src.data_postprocessing.certainty import calculate_certainty_and_uncertainty, get_classified_indices
from src.utils import get_base_dir, save_to_pickle, load_from_pickle, get_device
from src.utils_datasets import get_target_type, load_dataset, load_model

dataset_name = "mumin" #"politifact" #"mumin"
target_type = get_target_type(dataset_name)
split="60_15_25"
mode="EF_256"
seed=42
num_layers = 3 #2
#model_dir = os.path.join(get_base_dir(), f"best_models_{split}_{num_layers}layers")
model_dir = os.path.join(get_base_dir(), f"best_models_{split}_{num_layers}layers")
results_dir = os.path.join(get_base_dir(), "GNNExplainer")

topk = 50


if __name__ == "__main__":

    """
    data = load_dataset(dataset_name, split=split, mode=mode, seed=seed)
    #model, probs = load_model(dataset_name, model_dir=model_dir, model_type=mode, seed=seed, num_layers = num_layers, data=data)

    model_path = os.path.join(get_base_dir(), model_dir, f"mumin_{mode}_seed{seed}_model.pth")
    model = GAT(hidden_channels=64, out_channels=2, dropout=0.4, num_layers=3)
    model = to_hetero(model, data.metadata(), aggr="sum").to(get_device())
    model.load_state_dict(torch.load(os.path.join(get_base_dir(), model_path), weights_only=True))

    explanation = get_explanation(data, model, target_type) #entire dataset 
    contribution_scores = compute_cumulative_contributions(explanation)
    for key, value in contribution_scores.items():
        print(f"{key}: {value}")
    save_to_pickle(contribution_scores, os.path.join(get_base_dir(), "GNNExplainer", "contribution_scores.pkl"))
    """
    print("###################")
    contribution_scores = load_from_pickle(os.path.join(get_base_dir(), "GNNExplainer", "contribution_scores.pkl"))

    final_scores = compute_normalized_node_vs_edge_importance(contribution_scores)
    """
    for key, value in final_scores.items():
        print(f"{key}: {value}")
    """

    plot_nodes_vs_edges_importance(final_scores, os.path.join(get_base_dir(), "plots", "nodes_vs_edges_importance.pdf"))

    nodes_fname=os.path.join(get_base_dir(), "plots", "node_types_importance_log.pdf")
    plot_types_importance(contribution_scores, dataset_name, nodes_fname, nodes=True, log_scale=True)

    edges_fname = os.path.join(get_base_dir(), "plots", "edge_types_importance_log.pdf")
    plot_types_importance(contribution_scores, dataset_name, edges_fname, nodes=False, log_scale=True)

    """
    edge_importance = extract_edge_importance(contribution_scores)
    for key, value in edge_importance.items():
        print(f"{key}: {value}")
    node_importance = extract_node_importance(contribution_scores)
    for key, value in node_importance.items():
        print(f"{key}: {value}")
    """


    #plot_node_importance_only_topk(node_impotance, os.path.join(get_base_dir(), "plots", "node_importance_topk.pdf"))



    """
    ## Find indices where prediction is incorrect and ground truth is the tgt_class, sorted by uncertainty (ascending order)
    wrong_indices_0, probabilities0, predictions0, labels0 = get_classified_indices(probs, data[target_type].y, 0)
    print(f"Indices:{wrong_indices_0}")
    print(f"Probabilities:{probabilities0}")
    print(f"Predictions:{predictions0}")
    print(f"Labels:{labels0}")
    print("###############")
    wrong_indices_1, probabilities1, predictions1, labels1 = get_classified_indices(probs, data[target_type].y, 1)
    print(f"Indices:{wrong_indices_1}")
    print(f"Probabilities:{probabilities1}")
    print(f"Predictions:{predictions1}")
    print(f"Labels:{labels1}")
    """

    """
    certain_and_uncertain = calculate_certainty_and_uncertainty(probs, topk)
    save_to_pickle(certain_and_uncertain, os.path.join(results_dir, f"certain_and_uncertain_top{topk}.pkl"))
    for key, value in certain_and_uncertain.items():
        print(f"{key}: {value}")
    

    certain_and_uncertain = load_from_pickle(os.path.join(results_dir, f"certain_and_uncertain_top{topk}.pkl"))
    class_0_certain_ids_list = certain_and_uncertain["class_0_certain_ids"]
    class_0_certain_ids = torch.tensor(class_0_certain_ids_list)
    #explanation_class_0 = get_explanation(data, model, target_type, class_0_certain_ids)
    #mean_feature_importance = compute_feature_importance(explanation_class_0, target_type)
    #plot_feature_importance(mean_feature_importance)
    class_1_certain_ids_list = certain_and_uncertain["class_1_certain_ids"]
    class_1_certain_ids = torch.tensor(class_1_certain_ids_list)
    #explanation_class_1 = get_explanation(data, model, target_type, class_1_certain_ids)
    test_ids = torch.tensor(class_1_certain_ids[:3])
    explanation_test = get_explanation(data, model, target_type, test_ids)
    mean_edge_importance = compute_edge_importance(explanation_test)
    """






