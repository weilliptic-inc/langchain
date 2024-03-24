from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from typing import Any, List, Optional
from uuid import uuid4

DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_COLLECTION_METADATA = {"index": "DISKANN"}
DEFAULT_CONNECTION_ARGS = {
    "host": "localhost",
    "port": "8888",
}
DEFAULT_K = 5

class WAIDB(VectorStore):
    def __init__(
        self,
        embedding_function: Optional[Embeddings] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        connection_args: Optional[dict] = None,
    ) -> None:
        try:
            from waipy import Client

            # Initialize the embedding function
            self.embedding_function = embedding_function
            
            # Initialize client
            if connection_args is None:
                connection_args = DEFAULT_MILVUS_CONNECTION
            self.client = Client(connection_args["host"], connection_args["port"])

            # Create the collection
            if collection_metadata is None:
                collection_metadata = DEFAULT_COLLECTION_METADATA
            
            collection_params = {}
            collection_params['name'] = collection_name
            collection_params['dimensions'] = 384
            collection_params['index_type'] = "DISKANN"
            collection_params['metric_type'] = "L2"
            self.collection_name = collection_name
            self.collection_id = self.client.create_collection(params=collection_params)
            self.collection_context = {}
        except ImportError:
            raise ValueError(
                "Could not import waipy python package. "
            )
        return

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = self.embedding_function.embed_documents(texts)
        e_ids = [i for i in range(len(embeddings))]
        e_keys = [uuid4().hex for i in range(len(embeddings))]
        id_key_mappings = {}
        key_text_mappings = {}
        for e_id, e_key, text in zip(e_ids, e_keys, texts):
            id_key_mappings[e_id] = e_key
            key_text_mappings[e_key] = text

        self.client.insert(collection_id=self.collection_id, ids=e_ids, vectors=embeddings)
        self.client.build_index(collection_id=self.collection_id)
        
        # Update collection context
        self.collection_context["id_key_mappings"] = id_key_mappings
        self.collection_context["key_text_mappings"] = key_text_mappings
        return e_keys

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        query_embedding = self.embedding_function.embed_query(query)
        search_results = self.client.search_index(collection_id=self.collection_id, vectors=[query_embedding], k=DEFAULT_K)
        e_ids = search_results[0]
        e_keys = [self.collection_context["id_key_mappings"][e_id] for e_id in e_ids]
        docs: list[Document] = []
        for e_key in e_keys:
            text = self.collection_context["key_text_mappings"][e_key]
            docs.append(Document(page_content=text))
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[Embeddings] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: dict = DEFAULT_COLLECTION_METADATA,
        connection_args: dict = DEFAULT_CONNECTION_ARGS,
        **kwargs: Any,
    ) -> WAIDB:
        waidb = cls(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            **kwargs,
        )
        waidb.add_texts(texts)
        return waidb

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: Optional[Embeddings] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: dict = DEFAULT_COLLECTION_METADATA,
        connection_args: dict = DEFAULT_CONNECTION_ARGS,
        **kwargs: Any,
    ) -> WAIDB:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            connection_args=connection_args,
            **kwargs,
        )
