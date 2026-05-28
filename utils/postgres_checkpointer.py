"""PostgreSQL Checkpointer with async support."""

import asyncio
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


class AsyncPostgresSaver(BaseCheckpointSaver[str]):
    """Async wrapper for PostgresSaver using connection pool."""
    
    def __init__(self, saver: PostgresSaver):
        super().__init__()
        self.saver = saver
    
    @classmethod
    def from_conn_string(cls, conn_string: str, **kwargs):
        """Create from connection string using connection pool."""
        # 使用连接池而不是单一连接，这样可以保持连接开放
        # kwargs 参数用于传递连接选项（如 autocommit, prepare_threshold）
        pool = ConnectionPool(
            conn_string,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        
        # 创建 PostgresSaver 实例（使用连接池）
        saver = PostgresSaver(pool)
        return cls(saver)
    
    def setup(self):
        """Setup database tables."""
        self.saver.setup()
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.saver.get_tuple(config)
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async get tuple."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.saver.get_tuple, config)
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.saver.put(config, checkpoint, metadata, new_versions)
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async put."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.saver.put,
            config,
            checkpoint,
            metadata,
            new_versions
        )
    
    def list(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        return self.saver.list(config, filter=filter, before=before, limit=limit)
    
    async def alist(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        """Async list."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: list(self.saver.list(config, filter=filter, before=before, limit=limit))
        )
    
    def delete_thread(self, thread_id: str) -> None:
        self.saver.delete_thread(thread_id)
    
    async def adelete_thread(self, thread_id: str) -> None:
        """Async delete thread."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.saver.delete_thread, thread_id)
    
    def put_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = ""):
        return self.saver.put_writes(config, writes, task_id, task_path)
    
    async def aput_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = ""):
        """Async put writes."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.saver.put_writes,
            config,
            writes,
            task_id,
            task_path
        )
