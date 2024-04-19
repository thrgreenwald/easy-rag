from typing import List
from document import Document


def format_docs(docs: List[Document]) -> str:
    doc_strings = [doc.content for doc in docs]
    return "\n\n".join(doc_strings)
