import json
import asyncio
import uvicorn
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from agent.react_agent import ReactAgent
from agent.supervisor_agent import SupervisorAgent, CANCELLED_SIGNAL
from utils.cancellation import TaskCancelledError
from utils.logger_handler import logger
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):
    query: str
    session_id: str = ""
    user_id: str = "default_user"


# 单例模式：全局共享一个 SupervisorAgent 实例
supervisor_agent = SupervisorAgent()


@app.on_event("startup")
async def start_periodic_cleanup():
    """每 7 天跑一次全量记忆清理"""
    async def _run():
        while True:
            await asyncio.sleep(7 * 86400)
            try:
                from memory.memory_cleanup import cleanup
                from memory.mem0_service import mem0_service
                users = set()
                all_memories = mem0_service.get_all()
                items = all_memories.get("results") if isinstance(all_memories, dict) else all_memories
                if isinstance(items, list):
                    for item in items:
                        uid = item.get("user_id") or item.get("metadata", {}).get("user_id", "default_user")
                        users.add(uid)
                for uid in users:
                    cleanup(uid)
            except Exception:
                pass
    asyncio.create_task(_run())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions/messages/stream")
async def agent_service_stream(
        query: str = Query(None),
        body: StreamRequest = None,
        request: Request = None
):
    actual_query = query or (body.query if body else None)
    if not actual_query:
        raise HTTPException(status_code=400, detail="query is required")

    # 获取 session_id 和 user_id（由 Java 后端传递过来）
    session_id = body.session_id if body else ""
    user_id = body.user_id if body else "default_user"

    async def generate():
        first_chunk = True
        yield f"data: {json.dumps({'thinking': True})}\n\n"

        try:
            async for chunk in supervisor_agent.execute_stream(actual_query, session_id, user_id):
                # 检查客户端是否断开连接
                if request and await request.is_disconnected():
                    supervisor_agent.cancel_session(session_id)
                    logger.info(f"[SSE] 客户端断开连接 session={session_id}")
                    yield f"data: {json.dumps({'cancelled': True})}\n\n"
                    return

                # 检测取消信号哨兵值（使用 is 进行对象身份比较）
                if chunk is CANCELLED_SIGNAL:
                    yield f"data: {json.dumps({'cancelled': True})}\n\n"
                    return

                if first_chunk:
                    first_chunk = False
                    yield f"data: {json.dumps({'thinking': False})}\n\n"
                    await asyncio.sleep(0.1)

                # 将chunk拆分成更小的片段，实现真正的流式效果
                chunk_size = 5  # 每次发送5个字符
                for i in range(0, len(chunk), chunk_size):
                    small_chunk = chunk[i: i + chunk_size]
                    payload = json.dumps({"chunk": small_chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0.05)  # 控制发送速率

            yield f"data: {json.dumps({'done': True})}\n\n"
        except (asyncio.CancelledError, TaskCancelledError):
            # 捕获 asyncio.CancelledError（可能在 await sleep 时注入）
            # 和 TaskCancelledError（合作式取消异常）
            logger.info(f"[SSE] 任务取消 session={session_id}")
            yield f"data: {json.dumps({'cancelled': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    """取消指定会话的活跃流式任务。

    客户端（Java 后端或前端）调用此端点来取消正在执行的 Agent 任务。
    取消是合作式的：不会强制终止，而是在下一个 yield 检查点优雅退出。
    LangGraph 的 Checkpoint 状态在取消前已持久化，可以后续恢复。
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    cancelled = supervisor_agent.cancel_session(session_id)
    return {"success": cancelled, "session_id": session_id}


@app.delete("/sessions/{session_id}/memory")
async def delete_session_memory(session_id: str, user_id: str = "default_user"):
    """
    删除指定会话的记忆（checkpointer中的历史记录），删除前提取关键事实到长期记忆

    Args:
        session_id: 会话ID
        user_id: 用户ID（用于长期记忆）
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        await supervisor_agent.delete_session_memory(session_id, user_id)
        return {"success": True, "message": f"Session {session_id} memory deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session memory: {str(e)}")


@app.post("/rag/documents/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = Form("default_user")):
    try:
        contents = await file.read()
        from rag.upload_service import DocumentUploadService
        svc = DocumentUploadService()
        result = svc.upload(contents, file.filename, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
