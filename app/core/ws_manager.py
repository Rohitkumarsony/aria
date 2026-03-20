import json
from collections import defaultdict
from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self._clients: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        # Close any stale connections for this session before adding new one
        stale = list(self._clients.get(session_id, []))
        for old_ws in stale:
            try:
                await old_ws.close()
            except Exception:
                pass
        self._clients[session_id] = [ws]

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        conns = self._clients.get(session_id, [])
        if ws in conns:
            conns.remove(ws)

    async def send(self, session_id: str, event: str, data: dict) -> None:
        payload = json.dumps({"event": event, **data})
        dead = []
        for ws in list(self._clients.get(session_id, [])):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_id, ws)

    async def broadcast(self, event: str, data: dict) -> None:
        payload = json.dumps({"event": event, **data})
        for sid in list(self._clients):
            for ws in list(self._clients[sid]):
                try:
                    await ws.send_text(payload)
                except Exception:
                    self.disconnect(sid, ws)


ws_manager = WebSocketManager()