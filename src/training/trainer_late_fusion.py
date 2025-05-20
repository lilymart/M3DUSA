import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.utils import save_to_pickle, get_base_dir

""" TRAINING """

def train_fusion_model(fusion_model, train_loader, optimizer, criterion, device, loss_dir, n_epochs=50):
    fusion_model.to(device)
    fusion_model.train()
    loss_values = []
    loss_file = os.path.join(get_base_dir(), loss_dir, f"loss_values_fusion_{fusion_model.name}.pkl")

    for epoch in range(n_epochs):
        running_loss = 0.0

        for inputs1, inputs2, labels in train_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = fusion_model(inputs1, inputs2)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs1.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_values.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}')

    print('Fusion Model Training completed.')
    save_to_pickle(loss_values, loss_file)
    return fusion_model


def train_classifier(classifier, unique_embeddings, labels, optimizer, criterion, device, model_name, loss_dir, n_epochs=50):
    classifier.to(device)
    classifier.train()

    loss_values = []
    loss_file = os.path.join(get_base_dir(), loss_dir, f"loss_values_classifier_{model_name}.pkl")

    for epoch in range(n_epochs):
        running_loss = 0.0

        optimizer.zero_grad()
        logits = classifier(unique_embeddings)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * unique_embeddings.size(0)

        epoch_loss = running_loss / len(unique_embeddings)
        loss_values.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}')

    print('Classifier Training completed.')
    save_to_pickle(loss_values, loss_file)
    return classifier



""" EVALUATION"""
"""
def evaluate_fusion_model(fusion_model, train_loader, device):
    fusion_model.eval()

    all_fused_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs1, inputs2, labels in train_loader:

            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            fused_embedding = fusion_model(inputs1, inputs2)
            all_fused_embeddings.append(fused_embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_fused_embeddings = np.concatenate(all_fused_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_fused_embeddings, all_labels
"""


def evaluate_classifier(classifier, unique_embeddings, labels, criterion, device):
    classifier.eval()
    #unique_embeddings = torch.tensor(unique_embeddings).float().to(device)
    #labels = torch.tensor(labels).long().to(device)  # Use long for class labels in CrossEntropyLoss

    with torch.no_grad():

        logits = classifier(unique_embeddings)
        loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)

    # Convert predictions and probabilities to CPU and numpy arrays for metrics
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    probs = probs.cpu().numpy()[:, 1]  # Store the probability of the positive class #cpu().detach()

    # Compute metrics
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')

    auc = roc_auc_score(labels, probs)  # Binary case
    precision_0 = precision_score(labels, preds, pos_label=0)
    recall_0 = recall_score(labels, preds, pos_label=0)
    precision_1 = precision_score(labels, preds, pos_label=1)
    recall_1 = recall_score(labels, preds, pos_label=1)

    # Return loss and all metrics
    return f1_micro, f1_macro, f1_weighted, auc, precision_0, recall_0, precision_1, recall_1


def extract_fused_embeddings(fusion_model, full_loader, device, seed, embeddings_dir):
    all_fused_embeddings = []
    all_labels = []
    fusion_model.eval()

    with torch.no_grad():
        for inputs1, inputs2, labels in full_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            fused_embedding = fusion_model.encode(inputs1, inputs2)
            all_fused_embeddings.append(fused_embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_fused_embeddings = np.concatenate(all_fused_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(get_base_dir(), embeddings_dir, f'embeddings_{fusion_model.name}_seed_{seed}.npy'), all_fused_embeddings)

    return torch.tensor(all_fused_embeddings).float().to(device), torch.tensor(all_labels).long().to(device)




