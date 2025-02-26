import os
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv, find_dotenv

from data_preparation.df_transform import read_json_dataframe, basic_pipeline


def export_plotly_image(
        figure: go.Figure, output_path: str | None) -> None:
    if output_path:
        if output_path.endswith('html'):
            figure.write_html(output_path)
        else:
            figure.write_image(output_path)
    else:
        figure.show()


def define_width_and_height(
        output_path: str) -> Tuple[int | None, int | None]:
    """
    Define output image width and height according to output path
    """
    if not output_path:
        return None, None
    elif output_path.endswith('html'):
        return None, None
    return 1200, 800


def plot_2d_scatter(
        dataframe: pd.DataFrame,
        embedding_column: str,
        color_column: Optional[str] = None,
        title: Optional[str] = None,
        output_path: Optional[str] = None) -> None:
    width, height = define_width_and_height(output_path)
    fig = px.scatter(
        dataframe,
        x=f'{embedding_column}_x',
        y=f'{embedding_column}_y',
        color=color_column,
        width=width,
        height=height,
        title=title,
        custom_data=[
            'title',
            'tags_br',
        ],
        color_discrete_sequence=px.colors.qualitative.G10 + px.colors.qualitative.Vivid,
    )
    fig.update_traces(
        hovertemplate=(
            '<b>%{customdata[0]}</b><br><br>%{customdata[1]}'
        )
    )
    export_plotly_image(fig, output_path)


if __name__ == '__main__':

    load_dotenv(find_dotenv())

    df = read_json_dataframe(
        file_path=os.getenv('DATASET_PATH')
    )
    df = basic_pipeline(df)
    plot_2d_scatter(
        dataframe=df,
        embedding_column='tsne_fulltext',
        color_column='type_tag',
        output_path='../plots/scatter_2D_fulltext.html'
    )