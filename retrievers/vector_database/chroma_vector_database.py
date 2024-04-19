from typing import List
from document import Document
from retriever import Retriever


class ChromaVectorDatabase(Retriever):

    def retrieve_similar_docs(self, query, max_docs=5) -> List[Document]:
        print(f"query: {query}")
