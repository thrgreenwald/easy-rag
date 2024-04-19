from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from openai_generator import OpenAIGenerator
from retrievers.retriever import Retriever

generator_types = {
    "openai": OpenAIGenerator
}


class Generator(ABC, BaseModel):
    retriever: Retriever
    standalone: bool = True
    system_prompt: Optional[str] = None

    @abstractmethod
    def answer_user_question(self, query: str, chat_history: List[Tuple[str, str]]) -> str:
        pass


def get_generator(retriever: Retriever, generator_type: str = "openai"):
    generator_class = generator_types.get(generator_type)
    if generator_class:
        return generator_class(retriever=retriever)
    else:
        raise ValueError("Unsupported generator type.")
