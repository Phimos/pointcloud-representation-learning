import argparse

import pandas as pd
import scipy.stats
from sklearn.cluster import KMeans


def compute_entropy(series: pd.Series) -> float:
    """Compute entropy of a categorical series."""
    counts = series.value_counts()
    return scipy.stats.entropy(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=20)
    args = parser.parse_args()

    n_clusters = args.n_clusters

    embeddings = pd.read_csv("embeddings.csv")
    embeddings[["object_code", "grasping_pose"]] = (
        embeddings["code"].str.split("__", expand=True).rename(columns={0: "object_code", 1: "grasping_pose"})
    )
    embeddings = embeddings.set_index("code")

    kmeans = KMeans(n_clusters=20, random_state=0).fit(embeddings.filter(like="dim"))
    embeddings["cluster"] = kmeans.labels_

    # statistics = pd.read_csv("data/statistics.csv").set_index("code", drop=True)
    # statistics = pd.merge(embeddings, statistics, left_on="object_code", right_index=True, how="left")

    embedding_columns = embeddings.filter(like="dim").columns
    embeddings = embeddings.drop(columns=embedding_columns)
    embeddings.to_csv("cluster_20_info.csv")
