"""Development entry point: ``python -m bookrec``."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("bookrec.api:app", host="127.0.0.1", port=8000)
