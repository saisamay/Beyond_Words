from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Beyond Words")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).with_name("web"), html=True),
    name="web",
)
