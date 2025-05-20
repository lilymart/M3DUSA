import os
import sys
import torch
import pandas as pd
from torch_geometric.nn import to_hetero
import torch.nn as nn
import time

from src.models.GAT import GAT
from src.robustness_analysis.EF_robustness import masking_unimodal_text, masking_unimodal_network, masking_multimodal, \
    masking_multimodal_mixed
from src.training.trainer import train_node_classifier, eval_node_classifier
from src.utils import compute_weights, get_device, set_random_seed
from src.utils_datasets import get_target_type, load_dataset


#"$dataset_name" "$mode" "$seed_index" "$seed" "$embeddings_dir" "$models_dir" "$results_dir" "$losses_dir"
if __name__ == "__main__":

    mode = "EF_256"

    dataset_name = sys.argv[1]
    split = sys.argv[2]
    drop_perc = int(sys.argv[3])
    drop_method_name = sys.argv[4]
    run = int(sys.argv[5]) #seed_index
    seed = int(sys.argv[6])
    embeddings_dir = sys.argv[7]
    models_dir = sys.argv[8]
    results_dir = sys.argv[9]
    losses_dir = sys.argv[10]

    set_random_seed(seed)
    modality = f"drop_{drop_perc}_{drop_method_name}"

    # LOAD THE DATASET
    target_type = get_target_type(dataset_name)
    data = load_dataset(dataset_name, split=split, mode=mode, seed=seed)

    # UPDATE DATA BASED ON DROPPING
    if drop_method_name == "masking_unimodal_text":
        data = masking_unimodal_text(data, target_type, drop_perc)
    elif drop_method_name == "masking_unimodal_net":
        data = masking_unimodal_network(data, target_type, drop_perc)  # not_masked=True default
    elif drop_method_name == "masking_multimodal":
        data = masking_multimodal(data, target_type, drop_perc)
    else:  # drop_method_name == "masking_multimodal_mixed"
        data = masking_multimodal_mixed(data, target_type, drop_perc)

    # SET THE MODEL
    # model = None
    model = GAT(hidden_channels=64, out_channels=2, dropout=0.3, num_layers=3) #2 for politifact
    model = to_hetero(model, data.metadata(), aggr='sum')

    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)  # lr=0.005
    targets = data[target_type].y
    weights = compute_weights(targets).float().to(device)
    criterion = nn.CrossEntropyLoss(weights)

    # TRAIN THE MODEL
    start_time = time.time()
    model = train_node_classifier(model, data, optimizer, criterion, seed, target_type, embeddings_dir, losses_dir, modality, n_epochs=1000, patience=100, epsilon=1e-6)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')

    # EVALUATE THE MODEL
    f1_micro, f1_macro, f1_weigh, auc, prec_0, rec_0, prec_1, rec_1 = eval_node_classifier(model, data, target_type, seed, embeddings_dir, modality)

    print(f'f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, roc-auc: {auc:.3f}')
    print(f'precision_0: {prec_0:.3f}, recall_0: {rec_0:.3f},  precision_1: {prec_1:.3f}, recall_1: {rec_1:.3f}')

    # SAVE THE MODEL
    model_path = os.path.join(models_dir, f"{dataset_name}_{modality}_seed{seed}_model.pth")
    torch.save(model.state_dict(), model_path)

    # SAVE THE RESULTS
    df = pd.DataFrame([{
        'Seed': seed, 'F1_micro': f1_micro, 'F1_macro': f1_macro, 'ROC-AUC': auc, 'Prec_0': prec_0, 'Rec_0': rec_0,
        'Prec_1': prec_1, 'Rec_1': rec_1, 'Time': training_time
    }])
    results_path = os.path.join(results_dir, f'{dataset_name}_{modality}_seed{seed}_results.xlsx')
    print(df)
    df.to_excel(results_path, index=False)
    print(f"Saved at {os.path.abspath(results_path)}")



