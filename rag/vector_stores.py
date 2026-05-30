from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from utils.config_handler import get_abs_path
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_type,get_file_md5
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List


class VectorStoreService(object):
    def __init__(self):

       self.vector_store = Chroma(
           collection_name=chroma_conf["collection_name"],
           embedding_function = embedding_model,
           persist_directory = get_abs_path(chroma_conf["persist_directory"]),
       )

       self.spliter = RecursiveCharacterTextSplitter(
           chunk_size=chroma_conf["chunk_size"],
           chunk_overlap=chroma_conf["chunk_overlap"],
           separators=chroma_conf["separators"],
           length_function=len,
       )

       self._bm25_retriever = None

       self.load_document()

    def _init_bm25(self):
        if self._bm25_retriever is not None:
            return
        all_docs = self.vector_store.get(include=["documents"])
        documents = []
        if all_docs and all_docs.get("documents"):
            for i, content in enumerate(all_docs["documents"]):
                doc_id = all_docs.get("ids", [str(i)])[i] if all_docs.get("ids") else str(i)
                metadata = {}
                if all_docs.get("metadatas") and i < len(all_docs["metadatas"]):
                    metadata = all_docs["metadatas"][i] or {}
                documents.append(Document(page_content=content, metadata=metadata, id=doc_id))
        if documents:
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = chroma_conf["k"]
            logger.info(f"[VectorStore] BM25 retriever ready, docs: {len(documents)}")

    def _build_filter(self, user_id=None):
        if user_id:
            return {"$or": [{"user_id": user_id}, {"user_id": "__shared__"}]}
        return None

    def get_retriever(self, user_id=None):
        """
        获得单路向量库检索的retriever
        :param user_id:
        :return:
        """
        kwargs = {"k": chroma_conf["k"]}
        f = self._build_filter(user_id)
        if f:
            kwargs["filter"] = f
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def add_user_document(self, file_path: str, user_id: str):
        from utils.file_handler import txt_loader, pdf_loader

        filename = os.path.basename(file_path)

        # 检查该用户是否已上传过同名文件
        existing = self.vector_store.get(where={"$and": [{"user_id": user_id}, {"source": filename}]})
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            logger.info(f"[VectorStore] File {filename} already exists for user {user_id}, skip")
            return -1  # 返回 -1 表示文件已存在

        if file_path.endswith(".txt"):
            docs = txt_loader(file_path)
        elif file_path.endswith(".pdf"):
            docs = pdf_loader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        if not docs:
            return 0
        split_docs = self.spliter.split_documents(docs)
        for doc in split_docs:
            doc.metadata["user_id"] = user_id
            doc.metadata["source"] = filename
        self.vector_store.add_documents(split_docs)
        self._bm25_retriever = None
        logger.info(f"[VectorStore] Added {len(split_docs)} chunks from {file_path} for user {user_id}")
        return len(split_docs)

    def get_bm25_retriever(self):
        self._init_bm25()
        return self._bm25_retriever

    def hybrid_search(self, query: str, k: int = None, rrf_k: int = 60) -> List[Document]:
        """
        BM25 + vector retrieval + RRF fusion.

        RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each retriever i
        """
        if k is None:
            k = chroma_conf["k"]

        self._init_bm25()
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k * 2})

        vector_docs = vector_retriever.invoke(query)
        bm25_docs = self._bm25_retriever.invoke(query) if self._bm25_retriever else []

        doc_scores = {}
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.page_content
            doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items[:k]]

    def get_hybrid_retriever(self) -> BaseRetriever:
        """
        获得混合检索的retriever
        :return:
        """
        vs = self

        class HybridRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return vs.hybrid_search(query)

        return HybridRetriever()

    def load_document(self):

        def check_md5(md5_str: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), 'w', encoding="utf-8").close
                return False
            else:
                with open(get_abs_path(chroma_conf["md5_hex_store"]), 'r', encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == md5_str:
                            return True
                return False

        def save_md5(md5_str: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'a', encoding="utf-8") as f:
                f.write(md5_str + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_type"])
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5(path)
            if check_md5(md5_hex):
                logger.info(f"[load doc] file {path} already in store, skip")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[load doc] file {path} empty, skip")
                    continue
                split_documents: list[Document] = self.spliter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[load doc] file {path} empty after split, skip")
                for doc in split_documents:
                    doc.metadata["user_id"] = "__shared__"
                self.vector_store.add_documents(split_documents)
                save_md5(md5_hex)
                logger.info(f"[load doc] file {path} loaded into store")
            except Exception as e:
                logger.error(f"[load doc] file {path} load failed: {str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("my weight is 180 jin, size recommendation")
    print(res)
