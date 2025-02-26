import os
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.manifold import TSNE

from data_preparation.df_transform import read_json_dataframe


def tsne_dimensionality_reduction(
        dataframe: pd.DataFrame,
        embeddings_column: str,
        output_column: str,
        num_dim: Optional[int] = 2) -> pd.DataFrame:
    reducer = TSNE(
        n_components=num_dim
    )
    embeddings = dataframe[embeddings_column].to_list()
    reduced = reducer.fit_transform(np.array(embeddings))
    dataframe[f'{output_column}_x'] = [dim[0] for dim in reduced]
    dataframe[f'{output_column}_y'] = [dim[1] for dim in reduced]
    return dataframe


if __name__ == '__main__':

    load_dotenv(find_dotenv())

    df = read_json_dataframe(
        file_path=os.getenv('DATASET_PATH')
    )
    df = tsne_dimensionality_reduction(
        dataframe=df,
        embeddings_column='embeddings_fulltext',
        output_column='tsne_fulltext'
    )
    df.to_json(
        os.getenv('DATASET_PATH'),
        orient='table',
    )
