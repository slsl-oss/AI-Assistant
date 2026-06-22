import hashlib
import os.path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from markitdown import MarkItDown
from utils.logger_handler import logger


def get_file_md5(file_path: str) -> str:
    """传入文件路径，返回文件的md5值字符串"""
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件路径{file_path}不存在")
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]文件路径{file_path}不是文件")

    md5_obj = hashlib.md5()
    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算文件[file_path]的md5值失败，错误信息：{str(e)}")


def listdir_with_allowed_type(file_path: str, allowed_types: tuple[str]):
    """返回文件夹内的文件列表"""
    files = []
    if not os.path.isdir(file_path):
        logger.error(f"[listdir_with_allowed_type]文件路径{file_path}不是文件夹")
        return allowed_types
    for f in os.listdir(file_path):
        if f.endswith(allowed_types):
            files.append(os.path.join(file_path, f))
    return tuple(files)


def pdf_loader(file_path: str, passed=None) -> list[Document]:
    return PyPDFLoader(file_path).load()


def txt_loader(file_path: str) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()


def markitdown_loader(file_path: str) -> list[Document]:
    """使用 MarkItDown 将任意格式文档转为 Markdown，返回单个 Document"""
    md = MarkItDown()
    result = md.convert(file_path)
    text = result.text_content.strip()
    if not text:
        return []
    return [Document(
        page_content=text,
        metadata={"source": os.path.basename(file_path), "format": "markdown"}
    )]
