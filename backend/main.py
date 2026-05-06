from app.api.routes import router
from app.core.config import settings
from app.db.session import Base, engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Engineering Drawing Evaluation System",
    version="1.0.0",
    description="Computer vision based engineering drawing evaluation service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")
