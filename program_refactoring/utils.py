import re 
import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from uuid import uuid4
import os 

import sklearn
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx

def get_and_save_embeddings(queries, programs, ids, name, persist_directory, metadata=None, use_query=True):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ['OPENAI_API_KEY'],
                    model_name="text-embedding-ada-002"
                )
    if use_query:
        # index on the query
        docs = queries
    else:
        # index on the program
        docs = programs

    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory,chroma_db_impl="duckdb+parquet",))

    collection = chroma_client.get_or_create_collection(name=name,embedding_function=openai_ef)

    # create metadata
    metadatas = []
    if metadata is not None:
        assert(len(metadata) == len(queries))
        iterator = zip(queries, programs, metadata)
    else:
        iterator = zip(queries, programs)

    for item in iterator:
        if metadata is not None:
            query, program, metadata = item
            metadatas.append({"query": query, "program": program, "metadata": metadata})
        else:
            query, program = item
            metadatas.append({"query": query, "program": program})


    collection.add(documents=docs, ids=ids, metadatas=metadatas)

    chroma_client.persist()


def load_from_dir(persist_directory, name):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ['OPENAI_API_KEY'],
                    model_name="text-embedding-ada-002"
                )
    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory,chroma_db_impl="duckdb+parquet",))
    collection = chroma_client.get_or_create_collection(name=name,embedding_function=openai_ef)
    return collection


def cluster_embeddings(embeddings, ids):
    X = np.array(embeddings)
    clustering = AgglomerativeClustering(n_clusters = None, distance_threshold=0.1).fit(X)
    return clustering, ids

def create_graph(clustering, ids):
    graph = nx.DiGraph()

    # ids = all_data['ids']
    sample_size = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        for child_idx in merge:
            if child_idx < sample_size:
                # from docs: at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
                node_name = f"{child_idx}:{ids[child_idx]}"
                graph.add_node(node_name)
                graph.add_edge(i + sample_size, node_name)
            else:
                graph.add_edge(i + sample_size, child_idx )
    return graph 

def visualize_graph(graph): 
    lines = nx.generate_network_text(graph, with_labels=True)
    for line in lines:
        print(line)


def clean_header(header, code):
    len_head = len(header)
    code = code[len_head:].strip()
    return code