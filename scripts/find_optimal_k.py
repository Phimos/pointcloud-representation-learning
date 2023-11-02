import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer

embeddings = pd.read_csv("embeddings.csv")

embeddings[["object_code", "grasping_pose"]] = (
    embeddings["code"].str.split("__", expand=True).rename(columns={0: "object_code", 1: "grasping_pose"})
)
embeddings = embeddings.set_index("code")

visualizer = kelbow_visualizer(KMeans(random_state=0, n_init="auto"), embeddings.filter(like="dim"), k=(4, 30))
ax = visualizer.draw()
ax.get_figure().savefig("kelbow.png", dpi=300)
