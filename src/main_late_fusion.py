import os
import sys
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
import torch.nn as nn
import time

from src.data_loading.politifact_embeddings import get_reduced_embedding
from src.models.LF_attention import AttentionFusionModel
from src.models.LF_bilinear import BilinearFusionModel
from src.models.LF_equal_contribution import LateFusionModel
from src.models.LF_gated import GatedFusionModel
from src.models.LF_weighted import WeightedFusionModel
from src.models.classifier_enhanced import EnhancedClassifier
from src.training.trainer_late_fusion import train_fusion_model, train_classifier, evaluate_classifier, extract_fused_embeddings
from src.utils import compute_weights, get_device, set_random_seed
from src.utils_datasets import load_dataset, load_embeddings, get_target_type



if __name__ == "__main__":

    dataset_name = sys.argv[1]
    split = sys.argv[2]
    model_name = sys.argv[3]
    run = int(sys.argv[4])  # seed_index
    seed = int(sys.argv[5])
    embeddings_dir = sys.argv[6]
    models_dir = sys.argv[7]
    results_dir = sys.argv[8]
    losses_dir = sys.argv[9]

    set_random_seed(seed)

    emb_dim = 64
    n_epochs_fusion = 50
    n_epochs_classifier = 20
    batch_size = 32

    single_embedding_models_names = ["only_text_news", "only_net_edges_cl", "only_net_edges_mps_cl",
                                     "only_net_edges_meta_cl", "only_net_edges_mps_meta_cl", "EF_256_cl", "EF_all_cl"]

    LF_models_dict = {
        "LF_concat": LateFusionModel(embedding_dim=emb_dim, fusion_strategy='concat', name=model_name),
        "LF_avg_pool": LateFusionModel(embedding_dim=emb_dim, fusion_strategy='avg', name=model_name),
        "LF_max_pool": LateFusionModel(embedding_dim=emb_dim, fusion_strategy='max', name=model_name),
        "LF_weighted": WeightedFusionModel(embedding_dim=emb_dim, name=model_name),
        "LF_attention": AttentionFusionModel(embedding_dim=emb_dim, name=model_name),
        "LF_gated": GatedFusionModel(embedding_dim=emb_dim, name=model_name),
        "LF_bilinear": BilinearFusionModel(embedding_dim=emb_dim, name=model_name)
    }

    if model_name in single_embedding_models_names:
        LF_model = None
    else:
        LF_model = LF_models_dict[model_name]

    classifier = EnhancedClassifier(2 * emb_dim) if LF_model != None and LF_model.name == "LF_concat" else EnhancedClassifier(emb_dim)

    # LOAD THE DATASET
    data = load_dataset(dataset_name, split=split, mode="only_net_edges_mps", seed=seed)
    target_type = get_target_type(dataset_name)
    labels = data[target_type].y
    train_mask, val_mask, test_mask = data[target_type].train_mask, data[target_type].val_mask, data[target_type].test_mask
    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    weights = compute_weights(labels).float().to(device)

    start_time = time.time()

    if LF_model == None: #unimodal

        if "text" in model_name: #text
            e_text = torch.tensor(load_embeddings(dataset_name, embeddings_dir, model_name, seed), dtype=torch.float32)
            e_text = get_reduced_embedding(embedding=e_text, embedding_dim=emb_dim)
            unique_embeddings = e_text.to(device)
        else: #network
            e_net = torch.tensor(load_embeddings(dataset_name, embeddings_dir, model_name, seed), dtype=torch.float32)
            unique_embeddings = e_net.to(device)

    else: #multimodal

        e_text = torch.tensor(load_embeddings(dataset_name, embeddings_dir, "only_text_news", seed), dtype=torch.float32)
        e_text = get_reduced_embedding(embedding=e_text, embedding_dim=emb_dim)
        e_net = torch.tensor(load_embeddings(dataset_name, embeddings_dir, "only_net_edges_mps", seed), dtype=torch.float32)
        e_net, e_text, LF_model = e_net.to(device), e_text.to(device), LF_model.to(device)

        full_dataset_LF = TensorDataset(e_net, e_text, labels)
        full_loader_LF = DataLoader(full_dataset_LF, batch_size=batch_size, shuffle=False)

        train_dataset_LF = TensorDataset(e_net[train_mask], e_text[train_mask], labels[train_mask])
        val_dataset_LF = TensorDataset(e_net[val_mask], e_text[val_mask], labels[val_mask])
        test_dataset_LF = TensorDataset(e_net[test_mask], e_text[test_mask], labels[test_mask])

        train_loader_LF = DataLoader(train_dataset_LF, batch_size=batch_size, shuffle=True)
        val_loader_LF = DataLoader(val_dataset_LF, batch_size=batch_size, shuffle=False)
        test_loader_LF = DataLoader(test_dataset_LF, batch_size=batch_size, shuffle=False)

        fusion_optimizer = torch.optim.Adam(LF_model.parameters(), lr=0.005) #0.001
        fusion_criterion = nn.CrossEntropyLoss(weights)

        #TRAIN THE LATE FUSION MODEL
        LF_model = train_fusion_model(LF_model, train_loader_LF, fusion_optimizer, fusion_criterion, device, losses_dir, n_epochs_fusion)

        unique_embeddings, labels = extract_fused_embeddings(LF_model, full_loader_LF, device, seed, embeddings_dir)
        unique_embeddings, labels = torch.Tensor(unique_embeddings).to(device), torch.tensor(labels).to(device)

    train_dataset = TensorDataset(unique_embeddings[train_mask], labels[train_mask])
    val_dataset = TensorDataset(unique_embeddings[val_mask], labels[val_mask])
    test_dataset = TensorDataset(unique_embeddings[test_mask], labels[test_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005) #0.001
    classifier_criterion = nn.CrossEntropyLoss(weights)

    # TRAIN THE CLASSIFIER
    classifier = train_classifier(classifier, unique_embeddings[train_mask], labels[train_mask], classifier_optimizer, classifier_criterion, device, model_name, losses_dir, n_epochs_classifier)

    # EVALUATETHE CLASSIFIER
    f1_micro, f1_macro, f1_weigh, auc, prec_0, rec_0, prec_1, rec_1 = evaluate_classifier(classifier, unique_embeddings[test_mask], labels[test_mask], classifier_criterion, device)
    print(f'f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, f1-weighted: {f1_weigh:.3f}, roc-auc: {auc:.3f}')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time} seconds')

    # SAVE THE RESULTS
    df = pd.DataFrame([{
        'Seed': seed, 'F1_micro': f1_micro, 'F1_macro': f1_macro, 'ROC-AUC': auc, 'Prec_0': prec_0, 'Rec_0': rec_0,
        'Prec_1': prec_1, 'Rec_1': rec_1, 'Time': execution_time
    }])
    results_path = os.path.join(results_dir, f'{dataset_name}_{model_name}_seed{seed}_results.xlsx')
    print(df)
    df.to_excel(results_path, index=False)
    print(f"Saved at {os.path.abspath(results_path)}")
