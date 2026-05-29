import json
import asyncio
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from agent.react_agent import ReactAgent
from agent.supervisor_agent import SupervisorAgent
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions/messages/stream")
async def agent_service_stream(
        query: str = Query(None),
        body: StreamRequest = None
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

        async for chunk in supervisor_agent.execute_stream(actual_query, session_id, user_id):
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

    return StreamingResponse(generate(), media_type="text/event-stream")


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
