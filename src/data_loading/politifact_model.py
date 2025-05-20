import os
import torch
from torch_geometric.nn import to_hetero

from src.models.GAT import GAT
from src.utils import get_base_dir, get_device


def load_politifact_model(model_dir, model_type, seed, num_layers, data):
    model_path = os.path.join(get_base_dir(), model_dir, f"politifact_{model_type}_seed{seed}_model.pth")
    model = GAT(hidden_channels=64, out_channels=2, dropout=0.3, num_layers=num_layers)
    model = to_hetero(model, data.metadata(), aggr='sum')
    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x_dict, data.edge_index_dict)
        #pred = out["news"].argmax(dim=-1)
        probs = torch.softmax(out["news"], dim=-1)
    return model, probs

