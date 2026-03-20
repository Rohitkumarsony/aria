from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.core.config import get_settings
from app.core.dependencies import get_agent, get_retriever, get_validator, get_vector_store

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.sessions_dir).mkdir(parents=True, exist_ok=True)
    get_vector_store(); get_retriever(); get_agent(); get_validator()
    print("✓ Aria v2 ready")
    yield

app = FastAPI(title="Aria v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

import os
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
