import pandas as pd
import scipy.stats
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.cluster.silhouette import silhouette_visualizer

embeddings = pd.read_csv("embeddings.csv")
print(embeddings)

embeddings[["object_code", "grasping_pose"]] = (
    embeddings["code"].str.split("__", expand=True).rename(columns={0: "object_code", 1: "grasping_pose"})
)
embeddings = embeddings.set_index("code")
print(embeddings)

# # kelbow method
# visualizer = kelbow_visualizer(KMeans(random_state=0), embeddings.filter(like="dim"), k=(4, 30))
# # save figure
# ax = visualizer.draw()
# ax.get_figure().savefig("kelbow.png")


# visualizer = silhouette_visualizer(KMeans(10, random_state=0), embeddings.filter(like="dim"))
# print(visualizer.silhouette_score_)
# # save figure
# ax = visualizer.draw()
# ax.get_figure().savefig("silhouette.png")

kmeans = KMeans(n_clusters=20, random_state=0).fit(embeddings.filter(like="dim"))

embeddings["cluster"] = kmeans.labels_

print(embeddings)

mapping = pd.read_csv("data/category_mapping.csv")
mapping = mapping.set_index("code")

print("mapping")
print(mapping)

# embeddings["category"] = embeddings['object_code'].map(mapping["category"])

print(embeddings)
# print(embeddings.pivot_table(index="cluster", columns="category", aggfunc="size", fill_value=0))
# print(embeddings)


# cluster 0 / 4
# cluster 0 is better than 4
# new cluster entropy for old cluster ...
# small entropy -> more confident -> better cluster


statistics = pd.read_csv("data/statistics.csv").set_index("code", drop=True)

print(statistics)
print(embeddings)


statistics = pd.merge(embeddings, statistics, left_on="object_code", right_index=True, how="left")

print(statistics)


def compute_entropy(series: pd.Series) -> float:
    """Compute entropy of a categorical series."""
    counts = series.value_counts()
    return scipy.stats.entropy(counts)


# print(statistics.query("geo_label == 0"))
# print(statistics["geo_label"].unique())

# statistics["cluster"] = embeddings["cluster"]

print(compute_entropy(statistics.query("geo_label == 4")["category"]))
print(compute_entropy(statistics.query("geo_label == 0")["category"]))
# print(compute_entropy(statistics.query("geo_label == 1")["category"]))
# print(compute_entropy(statistics.query("geo_label == 3")["category"]))

print("-" * 80)
print(compute_entropy(statistics.query("geo_label == 4")["cluster"]))
print(compute_entropy(statistics.query("geo_label == 0")["cluster"]))
# print(compute_entropy(statistics.query("geo_label == 1")["cluster"]))
# print(compute_entropy(statistics.query("geo_label == 3")["cluster"]))

print(statistics)

statistics = statistics.drop(columns=[f"dim{i}" for i in range(1024)])
print(statistics)
statistics.to_csv("cluster_20_info.csv")


# # make tSNE plot for embeddings
# import plotly.express as px
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
# tsne_embedding = tsne.fit_transform(embeddings.filter(like="dim"))

# # can use mouse to highlight specific categories
# fig = px.scatter(
#     tsne_embedding,
#     x=0,
#     y=1,
#     color=embeddings["category"],
#     hover_data={"code": embeddings.index, "category": embeddings["category"]},
# )
# fig.update_layout(
#     title="tSNE Embeddings of PointNet Pretrained Model",
#     xaxis_title="tSNE Component 1",
#     yaxis_title="tSNE Component 2",
# )
# fig.write_html("pointnet_pretrain_tsne.html")
