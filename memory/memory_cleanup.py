from memory import memory_index
from memory.mem0_service import mem0_service
from utils.config_handler import memory_conf
from utils.logger_handler import logger


def _cfg():
    return {
        "limit": memory_conf.get("memory_limit_per_user", 100),
        "threshold": memory_conf.get("importance_threshold", 0.3),
        "retention_days": memory_conf.get("retention_days", 30),
    }


def cleanup_by_importance(user_id: str) -> int:
    hashes = memory_index.get_by_importance(user_id, _cfg()["threshold"])
    return _delete(hashes)


def cleanup_by_time(user_id: str) -> int:
    hashes = memory_index.get_by_time(user_id, _cfg()["retention_days"])
    return _delete(hashes)


def cleanup_by_capacity(user_id: str) -> int:
    cfg = _cfg()
    total = memory_index.count(user_id)
    if total <= cfg["limit"]:
        return 0
    items = memory_index.get_sorted_by_score(user_id)
    excess = total - cfg["limit"]
    hashes = [it["memory_hash"] for it in items[:excess]]
    return _delete(hashes)


def cleanup(user_id: str) -> dict:
    r = {}
    r["importance"] = cleanup_by_importance(user_id)
    r["time"] = cleanup_by_time(user_id)
    r["capacity"] = cleanup_by_capacity(user_id)
    if sum(r.values()):
        logger.info(f"[MemoryCleanup] user={user_id} {r}")
    return r


def _delete(hashes: list) -> int:
    if not hashes:
        return 0
    n = 0
    for h in hashes:
        if mem0_service.delete_by_hash(h):
            memory_index.remove(h)
            n += 1
    return n
