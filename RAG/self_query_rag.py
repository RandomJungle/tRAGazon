import os
from typing import List, Dict, Any

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI


def get_rating(rating: Dict[str, Any] | float, key: str) -> float:
    if isinstance(rating, dict):
        return rating.get(key, np.nan)
    return np.nan


def get_tag(tags_dict: Dict[str, List[str]] | float, tag_key: str) -> str:
    if isinstance(tags_dict, dict):
        relevant_tags = tags_dict.get(tag_key, [])
        if relevant_tags:
            return relevant_tags[0]
    return ''


def create_vector_database(
        dataframe: pd.DataFrame,
        text_column: str,
        embeddings_column: str,
        chroma_client: chromadb.Client) -> VectorStore:
    dataframe['metadata'] = dataframe.apply(
        lambda x: {
            'title': x['title'],
            'rating': get_rating(x['ratings'], 'rating'),
            'rating_count': get_rating(x['ratings'], 'count'),
            'type_tags': get_tag(x['tags'], 'type'),
            'cuisine_tags': get_tag(x['tags'], 'cuisine'),
            'ingredient_tags': get_tag(x['tags'], 'ingredient'),
            'meal_tags': get_tag(x['tags'], 'meal')
        },
        axis=1
    )
    collection = chroma_client.create_collection(name='recipes_db')
    collection.add(
        documents=dataframe[text_column].to_list(),
        metadatas=dataframe['metadata'].to_list(),
        embeddings=dataframe[embeddings_column].to_list(),
        ids=[str(idx) for idx in dataframe.index.tolist()]
    )
    langchain_chroma = Chroma(
        client=chroma_client,
        collection_name='recipes_db',
        embedding_function=MistralAIEmbeddings()
    )
    return langchain_chroma


def main(queries: List[str]) -> None:
    chroma_client = chromadb.Client()
    vector_store = create_vector_database(
        dataframe=df,
        text_column='fulltext',
        embeddings_column='embeddings_fulltext',
        chroma_client=chroma_client
    )
    model = ChatMistralAI(
        mistral_api_key=os.getenv('MISTRAL_API_KEY')
    )
    metadata_field_info = [
        AttributeInfo(
            name='title',
            description='The title of the recipe',
            type='str',
        ),
        AttributeInfo(
            name='rating',
            description='The recipe rating on a scale of 0 to 5',
            type='float',
        ),
        AttributeInfo(
            name='rating_count',
            description='The total number of ratings received by the recipe',
            type='int',
        ),
        AttributeInfo(
            name='type_tags',
            description='the types associated with the recipe, like soup, wrap, casserole, etc.',
            type='str',
        ),
        AttributeInfo(
            name='cuisine_tags',
            description='the geographical area and cuisine associated with the recipe',
            type='str',
        ),
        AttributeInfo(
            name='ingredient_tags',
            description='the most important ingredients of this recipe',
            type='str',
        ),
        AttributeInfo(
            name='meal_tags',
            description='the type of meal this recipe is prepared for, like dinner, lunch, etc.',
            type='str',
        ),
    ]
    document_content_description = 'Detailed cooking recipe'
    retrieval_prompt = get_query_constructor_prompt(
        document_contents=document_content_description,
        attribute_info=metadata_field_info
    )
    output_parser = StructuredQueryOutputParser.from_components(
        fix_invalid=True,
    )
    query_constructor = retrieval_prompt | model | output_parser
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vector_store,
        structured_query_translator=ChromaTranslator(),
        search_type='similarity'
    )
    for query in queries:
        response = retriever.invoke(query)
        print(response)


if __name__ == '__main__':

    load_dotenv(find_dotenv())
    df = pd.read_json(
        os.getenv('DATASET_PATH'), orient='table'
    )
    main(
        queries=[
            'Give me salmon recipes that do not use any herbs',
            'I want Casserole recipes that are rated above 4.5'
        ]
    )
