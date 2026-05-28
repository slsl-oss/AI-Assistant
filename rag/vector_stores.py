from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from utils.config_handler import get_abs_path
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_type,get_file_md5
from utils.logger_handler import logger
from langchain_core.documents import Document


"""
向量数据库服务类：负责向量数据库的创建，加载，检索等操作

"""
class VectorStoreService(object):
    def __init__(self):

       self.vector_store = Chroma(
           collection_name=chroma_conf["collection_name"],
           embedding_function = embedding_model,
           persist_directory = get_abs_path(chroma_conf["persist_directory"]),  # 使用绝对路径

       )

       self.spliter = RecursiveCharacterTextSplitter(
           chunk_size=chroma_conf["chunk_size"],  # 分割后每个文本段的最大长度
           chunk_overlap=chroma_conf["chunk_overlap"],  # 分割后每个文本段的重叠长度
           separators=chroma_conf["separators"],  # 自然段落划分符号的分隔符列表
           length_function=len,  # 计算文本长度的函数
       )  # 文本分割器对象

       # 自动加载文档到向量数据库
       self.load_document()


    def get_retriever(self):
        """
        :return: 放回向量检索器对象方便加入chain
        :param self:

         """
        return self.vector_store.as_retriever(search_kwargs={"k" : chroma_conf["k"]})


    def load_document(self):
        """
        :param self:
        :return:
        """

        def check_md5(md5_str: str):
            """
            检查传入的md5字符串是否已经被处理
            return False: 未处理过
            return True: 已处理过
            """
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # if能进入说明文件不存在 --> 肯定没有处理过
                open(get_abs_path(chroma_conf["md5_hex_store"]), 'w', encoding="utf-8").close  # 创建文件
                return False
            else:
                with open(get_abs_path(chroma_conf["md5_hex_store"]), 'r', encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == md5_str:
                            return True
                return False

        def save_md5(md5_str: str):
            """将传入的md5字符串，记录到文件中保存"""
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'a', encoding="utf-8") as f:
                f.write(md5_str + "\n")


        def get_file_documents(read_path:str):
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
            #获取文件MD5
            md5_hex = get_file_md5(path)

            if check_md5(md5_hex):
                logger.info(f"[加载知识库]文件{path}已经存在知识库中，跳过跳过")
                continue

            try:
                documents:list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]文件{path}内容为空，跳过跳过")
                    continue

                split_documents:list[Document] = self.spliter.split_documents(documents)

                if not split_documents:
                    logger.warning(f"[加载知识库]文件{path}内容分割后，为空，跳过跳过")

                #内容加载到数据库中
                self.vector_store.add_documents(split_documents)

                #记录处理过的md5值
                save_md5(md5_hex)

                logger.info(f"[加载知识库]文件{path}内容已经加载到知识库中")

            except Exception as e:
                #exc_info为True 会记录详细的报错信息， 如果为False，记录报错信息本身
                logger.error(f"[加载知识库]文件{path}内容加载到知识库中失败，错误信息：{str(e)}",exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()


    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)




