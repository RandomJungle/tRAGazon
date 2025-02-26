import json
import os
import time
import warnings
from io import BytesIO
from typing import Optional, List, Dict

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mistralai import Mistral, File, BatchJobOut
from tqdm import tqdm

from data_preparation.df_transform import chunk_dataframe, read_json_dataframe


def join_ingredient_list(
        ingredient_list: List[str] | float) -> str:
    if isinstance(ingredient_list, list):
        return '\n'.join(ingredient_list)
    return ''


def join_recipe_instructions(
        instructions_dict: Dict[str, str]) -> str:
    if isinstance(instructions_dict, dict):
        return '\n'.join(
            [
                f'{key}: {value}' for key, value
                in instructions_dict.items()
            ]
        )
    return ''


def create_text_column(
        dataframe: pd.DataFrame,
        column_name: Optional[str] = 'text') -> pd.DataFrame:
    dataframe[column_name] = dataframe.apply(
        lambda x: x['title'] + ', Ingredients: \n\n' +
                  join_ingredient_list(x['ingredients']) +
                  '\n\nRecipe: \n\n' + join_recipe_instructions(x['instructions']),
        axis=1
    )
    return dataframe


def query_embeddings(
        dataframe: pd.DataFrame,
        model_name: Optional[str] = 'mistral-embed',
        text_column: Optional[str] = 'text',
        num_chunks: Optional[int] = 1,
        output_column: Optional[str] = 'mistral_embeddings') -> pd.DataFrame:

    if not 'id' in dataframe.columns:
        dataframe['id'] = dataframe.index + 1

    data = dataframe[['id', text_column]]
    chunks = chunk_dataframe(data, num_chunks)
    outputs = []

    client = Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', '')
    )
    for chunk in tqdm(chunks):
        chunk_response = client.embeddings.create(
            model=model_name,
            inputs=chunk[text_column].tolist()
        )
        chunk[output_column] = [emb.embedding for emb in chunk_response.data]
        outputs.append(chunk)

    output_dataframe = pd.concat(outputs, ignore_index=True)
    merged = pd.merge(
        left=dataframe,
        right=output_dataframe[['id', output_column]],
        on='id',
        how='left',
        validate='1:1'
    )
    return merged


def print_stats(batch_job):
    """
    Print the statistics of the batch job.

    Args:
        batch_job: The batch job object containing job statistics.
    """
    succeeded = batch_job.succeeded_requests
    failed = batch_job.failed_requests
    total = batch_job.total_requests
    print(f"Total requests: {total}")
    print(f"Failed requests: {failed}")
    print(f"Successful requests: {succeeded}")
    print(f"Percent done: {round((succeeded + failed) / total, 4) * 100}")


def create_batch_embedding_file(
        dataframe: pd.DataFrame,
        model_name: str,
        num_chunks: int) -> BytesIO:

    buffer = BytesIO()
    if not 'id' in dataframe.columns:
        dataframe['id'] = dataframe.index + 1
    data = dataframe[['id', 'text']]
    chunks = chunk_dataframe(data, num_chunks)
    for index, chunk in enumerate(chunks):
        query = {
            'custom_id': str(index),
            'body': {
                'model': model_name,
                'inputs': chunk['text'].tolist()
            }
        }
        buffer.write(json.dumps(query).encode("utf-8"))
        buffer.write("\n".encode("utf-8"))
    return buffer


def run_batch_job_embeddings(
        batch_data,
        client: Mistral,
        model_name: str) -> BatchJobOut:

    batch_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model_name,
        endpoint='/v1/embeddings',
        metadata={
            'job_type': 'testing'
        }
    )

    while batch_job.status in ["QUEUED", "RUNNING"]:
        batch_job = client.batch.jobs.get(
            job_id=batch_job.id
        )
        print_stats(batch_job)
        time.sleep(1)

    print(f"Batch job {batch_job.id} completed with status: {batch_job.status}")
    return batch_job


def download_file(client, file_id, output_path):
    """
    Download a file from the Mistral server.

    Args:
        client (Mistral): The Mistral client instance.
        file_id (str): The ID of the file to download.
        output_path (str): The path where the file will be saved.
    """
    if file_id is not None:
        print(f"Downloading file to {output_path}")
        output_file = client.files.download(file_id=file_id)
        with open(output_path, "w") as f:
            for chunk in output_file.stream:
                f.write(chunk.decode("utf-8"))
        print(f"Downloaded file to {output_path}")


def query_batch_embeddings(
        dataframe: pd.DataFrame,
        error_path: str,
        success_path: str,
        model_name: Optional[str] = 'mistral-embed',
        num_chunks: Optional[int] = 1) -> None:

    client = Mistral(
        api_key=os.getenv('MISTRAL_API_KEY', '')
    )

    buffer_data = create_batch_embedding_file(
        dataframe=dataframe,
        model_name=model_name,
        num_chunks=num_chunks
    )

    batch_data = client.files.upload(
        file=File(
            file_name='file.jsonl',
            content=buffer_data.getvalue()
        ),
        purpose='batch'
    )

    batch_job = run_batch_job_embeddings(
        batch_data=batch_data,
        client=client,
        model_name=model_name
    )

    print(f"Job duration: {batch_job.completed_at - batch_job.created_at} seconds")

    download_file(client, batch_job.error_file, error_path)
    download_file(client, batch_job.output_file, success_path)


if __name__ == '__main__':

    warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

    load_dotenv(find_dotenv())

    df = read_json_dataframe(
        os.getenv('DATASET_PATH')
    )
    df = create_text_column(
        dataframe=df,
        column_name='fulltext'
    )

    new_df = query_embeddings(
        dataframe=df,
        text_column='fulltext',
        num_chunks=1000,
        output_column='embeddings_fulltext'
    )

    new_df.to_json(
        '/home/juliette/projects/tRAGazón/outputs/embeddings_full.json',
        orient='table'
    )