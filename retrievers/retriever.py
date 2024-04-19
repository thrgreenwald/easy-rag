from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel
from document import Document
from vector_index.faiss_vector_index import FAISSVectorIndex
from vector_database.chroma_vector_database import ChromaVectorDatabase
from langchain.embeddings import OpenAIEmbeddings

# shoutout langchain for their OpenAIEmbeddings class, I did NOT feel like rewriting this one
# source: https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.openai.OpenAIEmbeddings.html#langchain_community.embeddings.openai.OpenAIEmbeddings

retriever_types = {
    "faiss": FAISSVectorIndex,
    "chromadb": ChromaVectorDatabase
}

embedding_types = {
    "openai": OpenAIEmbeddings
}


class Retriever(ABC, BaseModel):
    # docs: List[Document]
    # split_docs: bool

    @abstractmethod
    def retrieve_similar_docs(self, query: str,
                              max_docs: int = 5) -> List[Document]:
        pass


def get_retriever(docs: List[Document], retriever_type: str = "faiss",
                  embedding_type: str = "openai", split_docs: bool = True,
                  language: Optional[str] = None, chunk_size: Optional[int] = None,
                  chunk_overlap: Optional[int] = None):
    retriever_class = retriever_types.get(retriever_type)
    if not retriever_class:
        raise ValueError(f"Unsupported retriever type: {retriever_type}")

    embedding_class = embedding_types.get(embedding_type)
    if not embedding_class:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    # Instantiate the embedder
    embedder = embedding_class()

    # Prepare additional keyword arguments for DocumentSplitter, if needed
    splitter_kwargs = {}
    if language is not None:
        splitter_kwargs["language"] = language
    if chunk_size is not None:
        splitter_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        splitter_kwargs["chunk_overlap"] = chunk_overlap

    # Call from_documents with the necessary arguments
    return retriever_class.from_documents(docs=docs, split_docs=split_docs, embedder=embedder, **splitter_kwargs)
