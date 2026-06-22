from abc import ABC,abstractmethod
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf, agent_conf

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def __init__(self, model_name: str = None):
        self._model_name = model_name or agent_conf["chat_model_name"]

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=self._model_name)

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 默认模型名（agent.yaml 的 chat_model_name，回退到 rag.yaml 的 chat_model_name）
_default_model_name = agent_conf.get("chat_model_name", rag_conf["chat_model_name"])


def _get_model_for_role(role: str):
    """根据角色名获取对应的模型实例。

    从 agent.yaml 的 agent_models 中查找角色对应的模型名，
    未配置时回退到默认模型名。
    """
    agent_models = agent_conf.get("agent_models", {})
    model_name = agent_models.get(role, _default_model_name)
    return ChatModelFactory(model_name).generator()


# 默认模型实例（兼容旧代码，供 rag_service、memory_scorer 等非Agent模块使用）
chat_model = ChatModelFactory(_default_model_name).generator()
embedding_model = EmbeddingsFactory().generator()

# Agent 角色专用模型实例
supervisor_model   = _get_model_for_role("supervisor")
summarizer_model   = _get_model_for_role("summarizer")
react_agent_model  = _get_model_for_role("react_agent")
research_agent_model = _get_model_for_role("research_agent")
editor_model       = _get_model_for_role("editor")
writer_model       = _get_model_for_role("writer")
reviewer_model     = _get_model_for_role("reviewer")
reviser_model      = _get_model_for_role("reviser")
publisher_model    = _get_model_for_role("publisher")
human_model        = _get_model_for_role("human")


if __name__ == '__main__':
    print(embedding_model)


