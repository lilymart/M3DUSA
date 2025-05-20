import torch
from torch_geometric.explain import Explainer, CaptumExplainer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def get_explanation(data, model, target_type, indices=None):

    if indices is None:
        indices = torch.arange(data.y_dict[target_type].shape[0])

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),  # InputXGradient
        explanation_type='phenomenon', # model's beahviour (model) vs individual predictions (phenomenon)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 menon',  # model's beahviour (model) vs individual predictions (phenomenon)
        node_mask_type='attributes',
        edge_mask_type='object', #None
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='probs', #probability of the positive class. ------  other types: log_probs, raw
        )
    )

    with torch.no_grad():
        explanation = explainer(
            data.x_dict,
            data.edge_index_dict,
            target=data.y_dict[target_type],
            index=indices
        )
        return explanation



def compute_cumulative_contributions(explanation):
    contribution_scores = {}

    # Node Feature Contributions
    for node_type, features in explanation.node_mask_dict.items():
        total_score = features.sum().item()
        num_instances = features.shape[0] * features.shape[1]  # Nodes * Features if we have features of different types
        weighted_score = total_score / num_instances if num_instances > 0 else 0
        mean_score = features.sum(-1).mean().item()

        contribution_scores[node_type] = {
            "cumulative_score": total_score,
            "weighted_score": weighted_score,
            "mean_score": mean_score,
            "topk_score": 0  # Placeholder
        }

    # Edge Contributions
    for edge_type, edges in explanation.edge_mask_dict.items():
        total_score = edges.sum().item()
        num_instances = edges.numel()  # Total number of edges in this type
        weighted_score = total_score / num_instances if num_instances > 0 else 0
        mean_score = edges.sum(-1).mean().item()

        contribution_scores[edge_type] = {
            "cumulative_score": total_score,
            "weighted_score": weighted_score,
            "mean_score": mean_score,
            "topk_score": 0  # Placeholder
        }

    # Compute top-k values
    min_nodes = min([features.shape[0] for features in explanation.node_mask_dict.values()])
    min_edges = min([edges.numel() for edges in explanation.edge_mask_dict.values()])
    topk = min(min_nodes, min_edges)

    for node_type, features in explanation.node_mask_dict.items():
        sorted_values = torch.sort(features.view(-1), descending=True).values
        contribution_scores[node_type]["topk_score"] = sorted_values[:topk].sum().item()

    for edge_type, edges in explanation.edge_mask_dict.items():
        sorted_values = torch.sort(edges.view(-1), descending=True).values
        contribution_scores[edge_type]["topk_score"] = sorted_values[:topk].sum().item()

    return contribution_scores


def compute_normalized_node_vs_edge_importance(contribution_scores):
    node_importance = {"cumulative": 0, "weighted": 0, "mean": 0, "topk": 0}
    edge_importance = {"cumulative": 0, "weighted": 0, "mean": 0, "topk": 0}

    # Sum up absolute contributions based on type (node vs. edge)
    for key, value in contribution_scores.items():
        category = node_importance if isinstance(key, str) else edge_importance

        category["cumulative"] += abs(value["cumulative_score"])
        category["weighted"] += abs(value["weighted_score"])
        category["mean"] += abs(value["mean_score"])
        category["topk"] += abs(value["topk_score"])

    # Normalize scores (sum to 1)
    for metric in ["cumulative", "weighted", "mean", "topk"]:
        total = node_importance[metric] + edge_importance[metric]
        if total > 0:
            node_importance[metric] /= total
            edge_importance[metric] /= total

    return {"node_importance": node_importance, "edge_importance": edge_importance}


def plot_nodes_vs_edges_importance(data, filename):

    node_importance = data['node_importance']
    edge_importance = data['edge_importance']

    strategies = ['cumulative', 'weighted', 'topk']
    new_labels = ['cumulative', 'mean', 'undersampling']

    node_values = [node_importance[strategy] for strategy in strategies]
    edge_values = [edge_importance[strategy] for strategy in strategies]

    x = np.arange(len(strategies))
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, node_values, width, label='Nodes importance', color='royalblue')
    ax.bar(x + width/2, edge_values, width, label='Edges importance', color='#A1CAF1')

    ax.set_xticks(x)
    ax.set_xticklabels(new_labels, fontsize=14, fontweight='bold')
    ax.set_xlabel("Aggregation strategy", fontsize=15, fontweight='bold')
    ax.set_ylabel('Importance score', fontsize=15, fontweight='bold')
    #ax.set_title('Node vs Edge Importance')
    ax.legend(fontsize=14)

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()



def extract_edge_importance(contribution_scores):
    edge_importance = {
        edge_type: [
            scores['cumulative_score'],
            scores['weighted_score'],
            scores['mean_score'],
            scores['topk_score']
        ]
        for edge_type, scores in contribution_scores.items()
        if isinstance(edge_type, tuple)  # Filtering only edges (which are stored as tuples)
    }
    return edge_importance


def extract_node_importance(contribution_scores):
    node_importance = {
        node_type: [
            scores['cumulative_score'],
            scores['weighted_score'],
            scores['mean_score'],
            scores['topk_score']
        ]
        for node_type, scores in contribution_scores.items()
        if isinstance(node_type, str)  # Filtering only nodes (which are stored as strings)
    }
    return node_importance


def processing_edge_importance_for_plot(data, dataset="politifact", pos=-1): #pos è la posizione del singolo valore che vogliamo plottare nella lista
    edges = extract_edge_importance(data)
    values = {key: value[pos] for key, value in edges.items()}
    if dataset == "politifact":
        processed_data = {
            "N-T": values[('tweet', 'discusses', 'news')] + values[('news', 'is_discussed_by', 'tweet')],
            "T-H": values[('tweet', 'has_hashtag', 'hashtag')] + values[('hashtag', 'is_hashtag_of', 'tweet')],
            "U-T_p": values[('user', 'posted', 'tweet')] + values[('tweet', 'is_posted_by', 'user')],
            "U-T_r": values[('user', 'retweeted', 'tweet')] + values[('tweet', 'is_retweeted_by', 'user')], #abs(values[('user', 'retweeted', 'tweet')])
            "U-U": values[('user', 'mentions', 'user')] + values[('user', 'is_mentioned_by', 'user')],
            "N-T-U-T-N": values[('news', 'metapath_0', 'news')],
            "N-T-H-T-N": values[('news', 'metapath_1', 'news')],
            "N-T-U-U-T-N": values[('news', 'metapath_2', 'news')]
        }
    else: #mumin
        processed_data = {
            "C-T": values[('tweet', 'discusses', 'claim')] + values[('claim', 'is_discussed_by', 'tweet')],
            "T-R_r": values[('tweet', 'is_replied_by', 'reply')] + values[('reply', 'reply_to', 'tweet')],
            "T-R_q": values[('tweet', 'is_quoted_by', 'reply')] + values[('reply', 'quote_of', 'tweet')],
            "T-H": values[('tweet', 'has_hashtag', 'hashtag')] + values[('hashtag', 'is_hashtag_of', 'tweet')],
            "T-A": values[('tweet', 'has_article', 'article')] + values[('article', 'is_article_of', 'tweet')],
            "T-I": values[('tweet', 'has_image', 'image')] + values[('image', 'is_image_of', 'tweet')],
            "U-T_m": values[('user', 'is_mentioned_by', 'tweet')] + values[('tweet', 'mentions', 'user')],
            "U-T_p": values[('user', 'posted', 'tweet')] + values[('tweet', 'is_posted_by', 'user')],
            "U-T_r": values[('user', 'retweeted', 'tweet')] + values[('tweet', 'is_retweeted_by', 'user')],
            # abs(values[('user', 'retweeted', 'tweet')])
            "U-R": values[('user', 'posted', 'reply')] + values[('reply', 'is_posted_by', 'user')],
            "U_U_f": values[('user', 'follows', 'user')] + values[('user', 'is_followed_by', 'user')],
            "U_U_m": values[('user', 'mentions', 'user')] + values[('user', 'is_mentioned_by', 'user')],
            "U-H": values[('user', 'has_hashtag', 'hashtag')] + values[('hashtag', 'is_hashtag_of', 'user')],
            "C-T-U-T-C": values[('claim', 'metapath_0', 'claim')],
            "C-T-H-T-C": values[('claim', 'metapath_1', 'claim')],
            "C-T-R-T-C_r": values[('claim', 'metapath_2', 'claim')],
            "C-T-R-T-C_q": values[('claim', 'metapath_3', 'claim')]
        }

    return processed_data


def processing_node_importance_for_plot(data, dataset="politifact", pos=-1): #pos è la posizione del singolo valore che vogliamo plottare nella lista
    nodes = extract_node_importance(data)
    values = {key: value[pos] for key, value in nodes.items()}

    if dataset == "politifact":
        processed_data = {
            "N": values["news"],
            "T": values["tweet"],
            "U": values["user"],
            "H": values["hashtag"]
        }
    else: #mumin
        processed_data = {
            "C": values["claim"],
            "T": values["tweet"],
            "U": values["user"],
            "H": values["hashtag"],
            "R": values["reply"],
            "I": values["image"],
            "A": values["article"]
        }

    return processed_data


def plot_barchart_importance(data, xlabel, color, bar_width, rotation, filename, log_scale=False):
    plt.figure(figsize=(10, 5))
    if log_scale:
        #plt.bar(data.keys(), np.log10(np.array(list(data.values())) + 1), color=color, width=bar_width) #TODO 10^0, 10^1 etc.
        values = np.array(list(data.values()))
        plt.bar(data.keys(), values, color=color, width=bar_width, log=True)
        plt.yscale("log")
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    else:
        plt.bar(data.keys(), data.values(), color=color, width=bar_width)
    plt.xlabel(xlabel, fontsize=15, fontweight='bold')
    plt.ylabel("Logarithmic importance score", fontsize=15, fontweight='bold')
    plt.xticks(rotation=rotation, fontsize=14, fontweight='bold')
    if filename != None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def plot_types_importance(contribution_scores, dataset, filename, nodes=True, log_scale=False):
    if nodes:
        data = processing_node_importance_for_plot(contribution_scores, dataset)
        xlabel="Node types"
        color = 'royalblue'
        bar_width = 0.4
        rotation=0
    else:
        data = processing_edge_importance_for_plot(contribution_scores, dataset)
        xlabel = "Edge and meta-path types"
        color = '#A1CAF1'
        bar_width = 0.8
        rotation=45
    plot_barchart_importance(data, xlabel, color, bar_width, rotation, filename, log_scale)


