# M3DUSA

Implementation of M3DUSA: a Modular Multi-Modal Deep fUSion Architecture for fake news detection on social media, as presented in our paper:
```
Martirano, L., Comito, C., Guarascio, M., Pisani, F.S. & Zicari, P.,
M3DUSA: A Modular Multi-Modal Deep fUSion Architecture for fake news detection on social media. 
SNAM (2025).
```

>Our proposed M3DUSA is multi-modal learning approach conceived to automatically identify deceptive or misleading news items by leveraging textual content, visual content and social network dynamics.
The objective is to classify a news item as factual or fake based on evidence extracted from the news-related content and its corresponding behavior on social media.

## Requirements
You can install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

Please note that some libraries (pyg-lib, torch-cluster, torch-scatter, torch-sparse and torch-spline-conv) may fail their installation.
You can install each of this library by running the following:
```
pip install library_name -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

## Data
To compile with [Twitter Developer Policy](https://developer.x.com/en/developer-terms/policy), Twitter datasets cannot be shared. 
For the PolitiFact dataset, you can follow the instructions in [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
For the MuMiN dataset, you can refer to [MuMiN](https://mumin-dataset.github.io/).


## Training and evaluation
For training M3DUSA in the early (late, resp.) fusion setting, run the script main.py (main_late_fusion.py) specifying the following parameters:
- *dataset_name* (e.g., "mumin" or "politifact")
- *split*, string specifying the train-validation-test split ratio (default is "60_15_25")
- *model*, string specifying the data loading process or the model that will feed the classifier.
   When running main.py, the allowed values are: "EF_all", "EF_256", "only_net_edges", "only_net_edges_mps", "only_net_edges_meta", "only_net_edges_mps_meta" (default is "EF_256"). 
   When running main_late_fusion.py, the allowed values are "LF_concat", "LF_avg_pool", "LF_max_pool", "LF_weighted", "LF_attention", "LF_gated", "LF_bilinear", and additionally: "only_text_news", "only_net_edges_cl", "only_net_edges_mps_cl", "only_net_edges_meta_cl", "only_net_edges_mps_meta_cl", "EF_256_cl", "EF_all_cl".
   See our paper for further details.
- *seed_index*, integer indicating the index of the seed to be used (in the current implementation any index can be specified)
- *seed*, integer indicating the seed to be used for reproducibility (default is 42)
- *embeddings_dir*, string indicating the absolute path where the embeddings are stored
- *models_dir*, string indicating the absolute path where the best model is stored
- *results_dir*, string indicating the absolute path where the results (evaluation metrics) are stored
- *losses_dir*, string indicating the absolute path where the loss values are stored

Note:
For training and evaluating the model, *run_experiments.sh* or *run_experiments_classifier.sh* can be modified with your data directory and your source code directory (the latter to be added to the PYTHONPATH environment variable).

## Reference
If you use this code, please cite our paper:

```
Martirano, L., Comito, C., Guarascio, M., Pisani, F.S. & Zicari, P.,
M3DUSA: A Modular Multi-Modal Deep fUSion Architecture for fake news detection on social media. 
SNAM (2025).
```


 


