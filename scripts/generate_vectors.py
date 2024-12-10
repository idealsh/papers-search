import pandas as pd
from sbert import sbert
from pathlib import Path

# The file is not tracked on git
df = pd.read_feather("./dataframes/papers-combined-ml.feather")

Path("generated").mkdir(exists_ok=True)
title_vec = sbert.transform(df.title).add_prefix("title_v_")
title_vec.to_feather("./generated/title_vec.feather")
print("Title done")
abstract_vec = sbert.transform(df.abstract).add_prefix("abstract_v_")
abstract_vec.to_feather("./generated/abstract_vec.feather")
print("Abstract done")

title_vec = pd.read_feather("./generated/title_vec.feather")
abstract_vec = pd.read_feather("./generated/abstract_vec.feather")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

TITLE_WEIGHT = 1 / 4
ABSTRACT_WEIGHT = 3 / 4

vectors = (
    title_vec.rename(columns=lambda x: x.removeprefix("title_")) * TITLE_WEIGHT
    + abstract_vec.rename(columns=lambda x: x.removeprefix("abstract_"))
    * ABSTRACT_WEIGHT
)
# similarity_mtx = pd.DataFrame(
#     cosine_similarity(vectors).astype(np.float16, copy=False),
#     index=vectors.index,
#     columns=vectors.index,
# )
sim = cosine_similarity(vectors)

top_5_arg = np.argpartition(sim, -6)[:, -6:]
top_5_data = np.take_along_axis(sim, top_5_arg, axis=1)

top_5_arg_argsorted = (-top_5_data).argsort()[:, 1:]
top_5_arg_sorted = np.take_along_axis(top_5_arg, top_5_arg_argsorted, axis=1)
top_5_data_sorted = np.take_along_axis(top_5_data, top_5_arg_argsorted, axis=1)

similar_indices = pd.Series(top_5_arg_sorted.tolist(), index=vectors.index).rename(
    "similar_indices"
)
similarity = pd.Series(top_5_data_sorted.tolist(), index=vectors.index).rename(
    "similarity"
)
pd.DataFrame([similar_indices, similarity]).transpose().to_feather(
    "./generated/similar_papers.feather"
)
print("Similarity matrix done")
