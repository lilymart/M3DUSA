from src.data_loading.mumin import load_mumin_heterodata
from src.data_loading.mumin_embeddings import load_mumin_embeddings
from src.data_loading.mumin_model import load_mumin_model
from src.data_loading.politifact import load_politifact_heterodata
from src.data_loading.politifact_embeddings import load_politifact_embeddings
from src.data_loading.politifact_model import load_politifact_model
from src.utils import get_base_dir


def load_dataset(dataset_name, split, mode, seed):
    base_dir = get_base_dir()
    if dataset_name == "mumin":
        return load_mumin_heterodata(base_dir, split, mode, seed)
    elif dataset_name == "politifact":
        return load_politifact_heterodata(base_dir, split, mode, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def get_target_type(dataset_name):
    if dataset_name == "mumin":
        return "claim"
    if dataset_name == "politifact":
        return "news"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def load_embeddings(dataset_name, emb_dir, emb_type, seed):
    if dataset_name == "mumin":
        return load_mumin_embeddings(emb_dir, emb_type, seed)
    if dataset_name == "politifact":
        return load_politifact_embeddings(emb_dir, emb_type, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def load_model(dataset_name, model_dir, model_type, seed, num_layers, data):
    if dataset_name == "mumin":
        return load_mumin_model(model_dir, model_type, seed, num_layers, data)
    if dataset_name == "politifact":
        return load_politifact_model(model_dir, model_type, seed, num_layers, data)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

