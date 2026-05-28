"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_stores import VectorStoreService
from utils.prompts_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()    # 提示词文本本身
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)  # 提示词模板
        self.model = chat_model
        self.chain = self.__int_chain()  # 总结链



    def __int_chain(self):

        def print_prompt(prompt):
            print("="*20)
            print(prompt.to_string())
            print("=" * 20)
            return prompt

        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()

        return chain


    #根据用户输入，在向量数据库中检索相关文档
    def retriever_docs(self, query :str) -> list[Document]:
        """
        根据用户输入，在向量数据库中检索相关文档
        :param query:
        :return:
        """
        return self.retriever.invoke(query)


    #rag :总结服务：用户提问，搜索参考资料，将提问和参考资料交给模型，让模型总结回复
    def rag_summarize(self, query :str) -> str:
        """
        总结服务：用户提问，搜索参考资料，将提问和参考资料交给模型，让模型总结回复
        :param query:
        :return:
        """
        context_docs: list[Document] = self.retriever_docs(query)


        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：内容：{doc.page_content} | 元数据：{doc.metadata}\n"


        return self.chain.invoke(
            {
                "input":query,
                "context":context
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("身高170cm,尺码推荐"))





