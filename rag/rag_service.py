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
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever(user_id=self.user_id)
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__int_chain()



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
    print("=== 测试1：无 user_id（查全库）===")
    rag1 = RagSummarizeService()
    docs = rag1.retriever_docs("身高170cm,尺码推荐")
    print(f"检索到 {len(docs)} 条")
    for d in docs:
        print(f"  user_id={d.metadata.get('user_id', 'N/A')} | {d.page_content[:60]}...")

    print("\n=== 测试2：user_id='default_user' ===")
    rag2 = RagSummarizeService(user_id="default_user")
    docs2 = rag2.retriever_docs("身高170cm,尺码推荐")
    print(f"检索到 {len(docs2)} 条")
    for d in docs2:
        print(f"  user_id={d.metadata.get('user_id', 'N/A')} | {d.page_content[:60]}...")

    if len(docs2) == 0:
        print("\n!!! context 为空原因：旧数据无 user_id metadata，需清理 chroma_db 重建")





