from pydantic import BaseModel
from typing import List, Optional
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from document import Document

# this file uses Langchain RecursiveCharacterTextSplitter under the hood

language_types = {
    "python": Language.PYTHON
}


class DocumentSplitter(BaseModel):
    language: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100

    def __init__(self, **data):
        super().__init__(**data)
        self.get_rcts_instance()

    def get_rcts_instance(self):
        if self.language:
            language_enum = language_types.get(self.language)
            if language_enum:
                self.rcts = RecursiveCharacterTextSplitter.from_language(language_enum, chunk_size=self.chunk_size,
                                                                         chunk_overlap=self.chunk_overlap)
            else:
                raise ValueError("Unsupported language type.")
        else:
            self.rcts = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)

    def convert_to_langchain_documents(self, documents: List[Document]) -> List[LangchainDocument]:
        langchain_documents = []
        for doc in documents:
            metadata = {'source': doc.source, 'title': doc.title}
            langchain_doc = LangchainDocument(
                page_content=doc.content, metadata=metadata)
            langchain_documents.append(langchain_doc)
        return langchain_documents

    def convert_to_documents(self, langchain_documents: List[LangchainDocument]) -> List[Document]:
        documents = []
        for langchain_doc in langchain_documents:
            content = langchain_doc.page_content
            source = langchain_doc.metadata['source']
            title = langchain_doc.metadata['title']
            doc = Document(content=content, source=source, title=title)
            documents.append(doc)
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        langchain_documents = self.convert_to_langchain_documents(documents)
        split_langchain_docs = self.rcts.split_documents(langchain_documents)
        split_docs = self.convert_to_documents(split_langchain_docs)
        print(f"Split {len(documents)} documents into {len(split_docs)} documents \
              based on chunk size {self.chunk_size} with overlap {self.chunk_overlap}.")
        return split_docs
