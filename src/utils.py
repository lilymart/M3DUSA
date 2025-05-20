import torch
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import glob
import re
from statistics import stdev
import scipy
from fontTools.ttx import process
from torch import nn


training_seed = [42, 123, 12345, 123123, 2025]

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_base_dir():
    return '/data'


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_clean_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the directory and all its contents
    os.makedirs(directory)


def load_from_pickle(pckl_file):
    file = open(pckl_file, 'rb')
    return pickle.load(file)


def save_to_pickle(data_dict, pckl_file):
    with open(pckl_file, 'wb') as file:
        pickle.dump(data_dict, file)


def get_sparse_eye(size):
    eye = scipy.sparse.eye(size)
    coo = eye.tocoo()
    values = coo.data
    indices = torch.LongTensor([coo.row, coo.col])
    i = torch.sparse.FloatTensor(indices, torch.FloatTensor(values), torch.Size([size, size]))
    return i

def learnable_embedding(tensor, embedding_dim):
    emb = nn.Embedding(tensor.shape[0], embedding_dim)
    return emb


def compute_weights(targets):
    total_samples = len(targets)
    class_counts = torch.bincount(targets) # Count the occurrences of each class (assuming classes are labeled as 0, 1, 2, ..., n-1)
    weights = total_samples / class_counts.float() # Compute weights inversely proportional to the class frequency
    weights /= weights.sum() ## Normalize the weights so they sum to 1 (optional)

    # Print information for debugging
    for i, count in enumerate(class_counts):
        print(f"Number of class {i}s: {count.item()}")
    print(f"Computed weights: {weights}")

    return weights

def merging_results_by_mode(dir):
    pattern = re.compile(r"mumin_(.*?)_seed\d+")
    grouped_files = {}
    for file in os.listdir(dir):
        if file.endswith(".xlsx"):
            match = pattern.search(file)
            if match:
                mode = match.group(1)  # Extract mode
                grouped_files.setdefault(mode, []).append(os.path.join(dir, file))

    # Process each mode group
    for mode, files in grouped_files.items():
        combined_df = pd.DataFrame()

        for file in files:
            df = pd.read_excel(file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        combined_df_processed = processing_results(combined_df)
        combined_df_processed.to_excel(os.path.join(dir, f"mumin_{mode}_results.xlsx"), index=False)


def processing_results(df):
    res = pd.DataFrame(columns=df.columns)
    # Compute mean and standard deviation
    for col in df.columns:
        lista = df[col].tolist()
        media, st_dev = process_metric(df, col)
        if col == "Seed":
            stringa = ""
        elif col == "Time":
            stringa = f"{round(media):d} ± {round(st_dev):d}"
        else:
            stringa = "{:0.3f}".format(media) + u"\u00B1" + "{:0.3f}".format(st_dev)
        lista.append(stringa)
        res[col] = lista
    return res


#compute mean and standard deviation for multiple runs
def process_metric(df, col):
    data = df[col]
    media = data.mean()
    st_dev = stdev(data.tolist())
    return media, st_dev


def merge_xlsx_files(dir):
    files = [f for f in os.listdir(dir) if f.endswith('.xlsx') and "seed" not in f]
    all_data = []
    for file in files:
        file_path = os.path.join(dir, file)
        df = pd.read_excel(file_path)
        last_row = df.iloc[-1].to_frame().T  # Convert Series to DataFrame
        filename = os.path.splitext(os.path.basename(file_path))[0]
        last_row["Approach"], last_row["Model"] = extract_approach_and_model(filename)
        all_data.append(last_row)

    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_excel(os.path.join(dir, "results_all_merged.xlsx"), index=False)


def extract_approach_and_model(filename):
    if "EF" in filename:
        approach = "M_EF"
        model = "in_feats_"+filename.split("_")[2]
    else:
        if "only_net" in filename:
            approach = "U_Net"
            substring = "only_net_"
        elif "only_text" in filename:
            approach = "U_Text" #"only_text_news"
            substring = "only_text_"
        else: #if "LF" in filename:
            approach = "M_LF"
            if "concat" in filename or "pool" in filename:
                approach += "_EC" #EC vs #AC --- LF.name --- politifact_{name}_results.xlsx #LF_max_pool, LF_attention
            else:
                approach += "_AC"
            substring = "LF_"
        index = filename.find(substring)
        index_start = index + len(substring)
        index_end = filename.find("_results")
        model = filename[index_start:index_end]
        if "pool" not in filename:
            model = model.replace("_", "+")
        if "cl" in filename:
            model = model.replace("+cl", " w/clf")
    return approach, model


def merge_xlsx_files_robustness_analysis(dir):
    files = [f for f in os.listdir(dir) if f.endswith('.xlsx') and "seed" not in f]
    all_data = []
    for file in files:
        file_path = os.path.join(dir, file)
        df = pd.read_excel(file_path)
        last_row = df.iloc[-1].to_frame().T  # Convert Series to DataFrame
        filename = os.path.splitext(os.path.basename(file_path))[0]
        last_row["Approach"], last_row["Model"] = extract_approach_and_model(filename)
        last_row["Drop"] = extract_drop_percentage(filename)
        all_data.append(last_row)

    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_excel(os.path.join(dir, "results_all_merged.xlsx"), index=False)


def extract_drop_percentage(filename):
    #if "drop" in filename:
    return filename.split("_")[2]


def processing_DPSG_results(directory, fname_in, fname_out):
    df = pd.read_excel(os.path.join(directory, fname_in))
    df_new = processing_results(df)
    df_new.to_excel(os.path.join(directory, fname_out), index=False)


def processing_HetSMCG_results(directory, fname_in, fname_out, method):
    df = pd.read_excel(os.path.join(directory, fname_in))
    df_sub = df[df["METHOD"] == method]
    df_sub.drop(columns=["METHOD"], inplace=True)
    df_new = processing_results(df_sub)
    df_new.to_excel(os.path.join(directory, fname_out), index=False)






if __name__ == "__main__":
    dataset_name = "politifact"  #"politifact" "mumin"

    #res_dir = os.path.join(get_base_dir(), 'robustness_analysis', "results_60_15_25_3layers") #'robustness_analysis',
    #merging_results_by_mode(res_dir) #STEP 0
    #merge_xlsx_files(res_dir) #STEP 1
    #merge_xlsx_files_robustness_analysis(res_dir) # STEP 1 alternativo in caso robustness analysis

    dir_competitors = f"/mnt/nas/martirano/{dataset_name}_DPSG_format"
    fname_in = f"GFN_result_{dataset_name}.xlsx"
    method="SAGE" #"GAT" "HGT" "SAGE"
    fname_out = f"HetSMCG_result_{method}_postprocessed.xlsx"
    processing_HetSMCG_results(dir_competitors, fname_in, fname_out, method)




