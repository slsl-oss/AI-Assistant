from abc import ABC,abstractmethod
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model= rag_conf["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model= rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()


if __name__ == '__main__':
    print(embedding_model)


