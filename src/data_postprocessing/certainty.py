import torch


def calculate_certainty_and_uncertainty(probs, top_k=50):
    # Most certain for class 0 (nodes where probability for class 0 is close to 0)
    class_0_certain_idx = torch.argsort(probs[:, 0])[top_k:] #maggiore precisione vicino allo 0

    # Most certain for class 1 (nodes where probability for class 1 is close to 1)
    class_1_certain_idx = torch.argsort(probs[:, 1])[top_k:] #maggiore precisione vicino allo 0

    # Most uncertain nodes (those with highest entropy)
    entropy = -probs[:, 0] * torch.log(probs[:, 0] + 1e-10) - probs[:, 1] * torch.log(probs[:, 1] + 1e-10)
    most_uncertain_idx = torch.argsort(entropy, descending=True)[:top_k]  # Sorting by entropy


    # Prepare the result dictionary with most certain and most uncertain nodes
    result_dict = {
        "class_0_certain_ids": class_0_certain_idx.tolist(),  # Top K most certain nodes for class 0
        "class_0_certain_probs": probs[class_0_certain_idx, 0].tolist(),  # Probabilities for class 0
        "class_1_certain_ids": class_1_certain_idx.tolist(),  # Top K most certain nodes for class 1
        "class_1_certain_probs": probs[class_1_certain_idx, 1].tolist(),  # Probabilities for class 1
        "most_uncertain_ids": most_uncertain_idx.tolist(),  # Top K most uncertain nodes
        "most_uncertain_entropy": entropy[most_uncertain_idx].tolist()  # Entropy for uncertain nodes
    }

    return result_dict


def get_classified_indices(probs, labels, tgt_class):
    predicted = torch.argmax(probs, dim=1) # Get predicted class (argmax of probs)

    # Find indices where prediction is incorrect and ground truth is the target_class
    misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]
    target_indices = (labels[misclassified_indices] == tgt_class).nonzero(as_tuple=True)[0]
    final_indices = misclassified_indices[target_indices]

    # Compute uncertainty as the absolute difference between the two probabilities
    uncertainty_scores = torch.abs(probs[final_indices, 0] - probs[final_indices, 1])

    # Sort indices by uncertainty (descending order)
    sorted_indices = final_indices[torch.argsort(uncertainty_scores)]
    sorted_probs = probs[sorted_indices]
    sorted_preds = predicted[sorted_indices]
    sorted_labels = labels[sorted_indices]

    return sorted_indices.tolist(), sorted_probs.tolist(), sorted_preds.tolist(), sorted_labels.tolist()



