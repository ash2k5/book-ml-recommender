"""Development entry point: ``python -m bookrec``."""

from bookrec.app import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
