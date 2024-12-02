from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
from pathlib import Path

from api.routes import router
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_app() -> FastAPI:
    app = FastAPI()
    
    # Get the absolute path to the static directory
    current_dir = Path(__file__).parent
    static_dir = current_dir / "static"
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Add routes
    app.include_router(router)
    
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)