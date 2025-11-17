from typing import Dict, Any, List
import os
from transcribe import transcribe_audio
from rag import embed_and_store, search_rag
from db import (
    list_documents,
    insert_document,
    add_telegram_group,
    list_telegram_groups,
    delete_telegram_group,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def rag_tool(query: str) -> Dict[str, Any]:
    """Run a RAG search for the query and return results."""
    try:
        results = search_rag(query, k=5)
        return {"tool": "rag", "query": query, "results": results}
    except Exception as e:
        # fallback to listing recent docs
        docs = list_documents(limit=5)
        return {"tool": "rag", "query": query, "error": str(e), "fallback_docs": docs}


def db_tool(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Basic DB tool stub. action can be 'insert_document' or other CRUD operations."""
    if action == "insert_document":
        title = payload.get("title")
        content = payload.get("content")
        metadata = payload.get("metadata")
        embedding = payload.get("embedding")
        doc_id = insert_document(title=title, content=content, metadata=metadata, embedding=embedding)
        return {"tool": "db", "action": action, "doc_id": doc_id}
    return {"tool": "db", "action": action, "payload": payload}


def telegram_tool(command: str, group_id: int = None, name: str = None, description: str = None) -> Dict[str, Any]:
    """Manage telegram groups stored in DB. This is an admin tool; actual bot actions are separate.

    Commands: 'add', 'list', 'delete'
    """
    if command == "add":
        if group_id is None or name is None:
            return {"ok": False, "error": "group_id and name required to add"}
        gid = add_telegram_group(tg_id=group_id, name=name, description=description or "")
        return {"ok": True, "group_db_id": gid}
    if command == "list":
        groups = list_telegram_groups()
        return {"ok": True, "groups": groups}
    if command == "delete":
        if group_id is None:
            return {"ok": False, "error": "group_id required to delete"}
        ok = delete_telegram_group(group_id)
        return {"ok": ok}
    return {"ok": False, "error": "unknown command"}


def route_text(text: str) -> Dict[str, Any]:
    """Heuristic router: direct text to DB, Telegram, or RAG.

    - customer/account/update/delete -> DB tool
    - group/telegram/chat -> Telegram tool
    - otherwise -> RAG
    """
    t = (text or "").lower()
    if any(k in t for k in ("customer", "account", "update", "delete", "create")):
        return db_tool("query", {"text": text})
    if any(k in t for k in ("group", "telegram", "chat")):
        return telegram_tool("list")
    return rag_tool(text)


if __name__ == "__main__":
    print("agent_router module - call route_text(text)")
