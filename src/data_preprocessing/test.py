import os
import pandas as pd

dataset = "gossipcop" # politifact gossipcop
base_dir = f"/mnt/nas/martirano/{dataset}_imgs"
input_dir = os.path.join(base_dir, "original_data")
output_dir = os.path.join(base_dir, "heterodata")

node_types = ["news", "user", "tweet", "reply"]
target_type = node_types[0]
edge_types = ["tweet_discusses_news", "user_likes_tweet", "user_posted_tweet", "user_retweeted_tweet", "user_posted_reply",
              "reply_is_reply_of_tweet", "user_mentions_user"]



if __name__ == "__main__":
    """
    for node_type in node_types:
        df = pd.read_csv(os.path.join(input_dir, "nodes", f"{node_type}.csv"))
        print(node_type)
        print(df.shape)
        print(df.columns)
        print("#############")

    for edge_type in edge_types:
        df = pd.read_csv(os.path.join(input_dir, "edges", f"{edge_type}.csv"))
        print(edge_type)
        print(df.shape)
        print(df.columns)
        print("#############")
    """

    df = pd.read_csv(os.path.join(input_dir, "nodes", "news.csv"))
    print("")


