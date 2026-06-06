"""WSGI entry point for production servers: ``gunicorn wsgi:app``."""

from bookrec.app import create_app

app = create_app()
