import os
from typing import List

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Output
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI


def create_vector_database(
        dataframe: pd.DataFrame,
        text_column: str,
        embeddings_column: str) -> VectorStore:
    dataframe['metadata'] = dataframe.apply(
        lambda x: {
            'title': x['title'],
            'rating': x['ratings'],
            'tags': x['tags']
        },
        axis=1
    )
    text_embedding_pairs = zip(
        dataframe[text_column].to_list(),
        dataframe[embeddings_column].to_list()
    )
    vector = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=MistralAIEmbeddings(),
        metadatas=dataframe['metadata'].tolist()
    )
    return vector


def create_vector_retriever(
        retriever: VectorStoreRetriever) -> Runnable:
    model = ChatMistralAI(
        mistral_api_key=os.getenv('MISTRAL_API_KEY')
    )
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on the provided context:

        <context>
        {context}
        </context>

        Question: {input}"""
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def query_vector_database(
        retrieval_chain: Runnable,
        input_question: str) -> Output:
    response = retrieval_chain.invoke(
        {
            'input': input_question
        }
    )
    return response


def run_rag_queries(
        vector_db: VectorStore,
        queries: List[str]) -> None:
    db_retrieval_chain = create_vector_retriever(
        retriever=vector_db.as_retriever()
    )
    for query in queries:
        response = query_vector_database(
            retrieval_chain=db_retrieval_chain,
            input_question=query
        )
        print('*' * 100)
        print(response['input'])
        print('---->')
        print(response['answer'])


def similarity_search(
        vector_db: VectorStore,
        query: str) -> None:
    results = vector_db.similarity_search_with_score(
        query=query,
        k=5,
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.metadata}")


if __name__ == '__main__':

    load_dotenv(find_dotenv())
    df = pd.read_json(
        os.getenv('DATASET_PATH'), orient='table'
    )
    vector_store = create_vector_database(
        dataframe=df,
        text_column='fulltext',
        embeddings_column='embeddings_fulltext'
    )
    similarity_search(
        vector_db=vector_store,
        query='Recipe with salmon and walnuts'
    )
