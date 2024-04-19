from pydantic import BaseModel
from typing import Optional, Dict


class Document(BaseModel):
    content: str
    title: Optional[str] = None
    source: Optional[str] = None


class DocStore(BaseModel):
    # {UUID -> Document}
    mapping: Dict[str, Document] = {}

    def add(self, mapping: Dict[str, Document]):
        overlap = set(mapping).intersection(self.mapping)
        if overlap:
            raise ValueError(f"IDs already exist in map: {overlap}")
        self.mapping = {**self.mapping, **mapping}

    def search(self, id):
        return self.mapping[id]
