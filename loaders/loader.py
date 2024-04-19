from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from document import Document


class Loader(ABC, BaseModel):
    # system_prompt: Optional[str] = None

    @abstractmethod
    def load_documents(self) -> List[Document]:
        pass


# def get_generator(retriever: Retriever, generator_type: str = "openai"):
#     generator_class = generator_types.get(generator_type)
#     if generator_class:
#         return generator_class(retriever=retriever)
#     else:
#         raise ValueError("Unsupported generator type.")
