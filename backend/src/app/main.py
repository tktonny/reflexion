"""FastAPI application entrypoint for the clinic diagnostic-adjunct service."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.src.app.api.routes import build_api_router
from backend.src.app.services.assessment_service import ClinicAssessmentService
from clinic.configs.settings import get_settings


settings = get_settings()
service = ClinicAssessmentService(settings)
app = FastAPI(title=settings.app_name)


def configure_app_logging() -> None:
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    for logger_name in (
        "clinic",
        "clinic.api",
        "clinic.realtime",
        "clinic.identity",
    ):
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(logging.INFO)
        if uvicorn_error_logger.handlers:
            target_logger.handlers = uvicorn_error_logger.handlers.copy()
            target_logger.propagate = False


configure_app_logging()

repo_dir = Path(__file__).resolve().parents[3]
clinic_frontend_src_dir = repo_dir / "clinic" / "frontend_app" / "src"
clinic_pages_dir = clinic_frontend_src_dir / "pages"
doctor_frontend_src_dir = repo_dir / "doctor" / "frontend_app" / "src"
doctor_pages_dir = doctor_frontend_src_dir / "pages"
care_frontend_src_dir = repo_dir / "care" / "frontend_app" / "src"
care_pages_dir = care_frontend_src_dir / "pages"

app.include_router(build_api_router(service), prefix="/api")
app.mount("/static", StaticFiles(directory=clinic_frontend_src_dir), name="clinic-static")
app.mount("/doctor/static", StaticFiles(directory=doctor_frontend_src_dir), name="doctor-static")
app.mount("/care/static", StaticFiles(directory=care_frontend_src_dir), name="care-static")


def page_response(name: str) -> FileResponse:
    return FileResponse(clinic_pages_dir / name)


def doctor_page_response(name: str) -> FileResponse:
    return FileResponse(doctor_pages_dir / name)


def care_page_response(name: str) -> FileResponse:
    return FileResponse(care_pages_dir / name)


@app.get("/")
@app.get("/clinic")
@app.get("/mirror")
async def clinic_index() -> FileResponse:
    return page_response("index.html")


@app.get("/freetalk")
@app.get("/clinic/freetalk")
async def freetalk_index() -> FileResponse:
    return page_response("freetalk.html")


@app.get("/doctor")
async def doctor_index() -> FileResponse:
    return doctor_page_response("index.html")


@app.get("/care")
@app.get("/caregiver")
async def caregiver_index() -> FileResponse:
    return care_page_response("index.html")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "backend.src.app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
    )
