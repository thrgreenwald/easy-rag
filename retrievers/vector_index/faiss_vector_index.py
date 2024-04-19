import faiss
import numpy as np
import uuid
import pickle
from pathlib import Path
from typing import List, Any, Dict
from document import Document, DocStore
from retriever import Retriever
from text_splitters import DocumentSplitter


class FAISSVectorIndex(Retriever):
    embedder: Any
    docstore: DocStore
    index: Any
    index_to_docstore_id: Dict[int, str]

    def retrieve_similar_docs(self, query, max_docs=5) -> List[Document]:
        embedding = self.embedder.embed_query(query)
        vector = np.array([embedding], dtype=np.float32)
        scores, indices = self.index.search(vector, max_docs)
        docs_and_scores = []
        for i, idx_id in enumerate(indices[0]):
            if idx_id == -1:
                # This happens when not enough docs are returned.
                continue
            ds_id = self.index_to_docstore_id[idx_id]
            doc = self.docstore.search(ds_id)
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Could not find document for id {ds_id}, got {doc}")
            docs_and_scores.append((doc, scores[0][i]))

        # scores can be used later
        return [doc for doc, _ in docs_and_scores[:max_docs]]

    @classmethod
    def from_documents(cls, docs, split_docs, embedder, **kwargs) -> 'FAISSVectorIndex':
        if split_docs:
            ds = DocumentSplitter(**kwargs)
            docs = ds.split_documents(docs)
        texts = [doc.content for doc in docs]
        embeddings = embedder.embed_documents(texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))

        inst = cls(embedder=embedder, docstore=DocStore(), index=index, index_to_docstore_id={})
        inst.add_to_index(docs, texts, embeddings)

        return inst

    def add_to_index(self, documents, texts, embeddings) -> List[str]:
        # Add to the index.
        vector = np.array(embeddings, dtype=np.float32)
        # if self._normalize_L2:
        #     faiss.normalize_L2(vector)
        self.index.add(vector)

        # Add information to docstore and index.
        ids = [str(uuid.uuid4()) for _ in texts]
        self.docstore.add({_id: doc for _id, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + i: _id for i, _id in enumerate(ids)}
        self.index_to_docstore_id.update(index_to_id)
        return ids

    def save_local(self, folder_path) -> None:
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss.write_index(self.index, str(path / "faiss_index.faiss"))

        # save docstore and index_to_docstore_id
        with open(path / "faiss_index.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(cls, folder_path, embedder) -> 'FAISSVectorIndex':
        path = Path(folder_path)

        # load index separately since it is not picklable
        index = faiss.read_index(str(path / "faiss_index.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / "faiss_index.pkl", "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)

        return cls(embedder=embedder, docstore=docstore, index=index,
                   index_to_docstore_id=index_to_docstore_id)
