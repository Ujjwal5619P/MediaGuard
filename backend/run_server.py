"""Run the API on Render's injected port, or 8000 for local development."""

import os

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=os.getenv("MEDIAGUARD_HOST", settings.host),
        port=int(os.getenv("PORT", str(settings.port))),
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
