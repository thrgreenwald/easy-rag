from typing import List, Tuple, Optional
from document import Document
from loader import Loader


class GenericTextLoader(Loader):
    content: List[str]

    def load_documents(self) -> List[Document]:
        print("documents")

