from pydantic import BaseModel
from typing import Optional, NamedTuple, List, Literal, Any, Tuple
from retrievers.retriever import Retriever
from generators.generator import Generator


class RAGSession(BaseModel):
    retriever: Retriever
    generator: Generator
    chat_history: List[Tuple[str, str]] = []

# TODO: rethink the notion of a RAG Session
