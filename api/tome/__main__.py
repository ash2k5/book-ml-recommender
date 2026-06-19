"""Development entry point: ``python -m tome``."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("tome.api:app", host="127.0.0.1", port=8000)
