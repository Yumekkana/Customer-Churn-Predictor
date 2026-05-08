import os

from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse


load_dotenv()

API_KEY = os.getenv("API_KEY")


async def api_key_middleware(request: Request, call_next):
    public_paths = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    if request.url.path in public_paths:
        return await call_next(request)

    api_key = request.headers.get("x-api-key")

    if not API_KEY:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "API key is not configured on the server."
            },
        )

    if api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Invalid or missing API key."
            },
        )

    return await call_next(request)