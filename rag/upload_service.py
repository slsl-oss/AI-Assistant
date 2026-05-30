import os
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from rag.vector_stores import VectorStoreService


class DocumentUploadService:

    def __init__(self):
        self.vector_store = VectorStoreService()

    def upload(self, file_bytes: bytes, filename: str, user_id: str) -> dict:
        upload_dir = get_abs_path(f"data/uploads/{user_id}")
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"[Upload] Saved file: {file_path}")

        try:
            chunk_count = self.vector_store.add_user_document(file_path, user_id)
            if chunk_count == -1:
                return {"success": False, "filename": filename, "error": "文件已存在，请勿重复上传"}
            return {"success": True, "filename": filename, "chunks": chunk_count}
        except Exception as e:
            logger.error(f"[Upload] Failed to index file: {e}")
            return {"success": False, "filename": filename, "error": str(e)}
