# #Copyright (C) 2025 β ORI Inc.
# #Written by Awase Khirni Syed 2025
import os
from typing import Dict, List

# Project configuration
PROJECT_DIR = "project"
DATABASE_URL = "postgresql+psycopg2://awase:password@localhost:5432/db"

# Define folder structure with optional files per folder
FOLDER_STRUCTURE = {
    # Core
    "domain": ["__init__.py"],
    "application": ["__init__.py"],
    
    # Ports
    "ports/http/controllers": ["__init__.py"],
    "ports/http/dtos": ["__init__.py"],
    "ports/http/mappers": ["__init__.py"],
    "ports/http/validations": ["__init__.py"],
    "ports/database": ["__init__.py"],

    # Adapters
    "adapters/database": ["__init__.py", "db.py"],
    "adapters/database/repositories": ["__init__.py"],
    "adapters/http/controllers": ["__init__.py"],
    "adapters/http/dtos": ["__init__.py"],
    "adapters/http/mappers": ["__init__.py"],
    "adapters/http/validations": ["__init__.py"],

    # Shared utilities
    "shared/utils": ["__init__.py", "logger.py", "exceptions.py", "helpers.py"],

    # Config
    "config": ["__init__.py", "config.py", "dependencyinjection.py"],

    # Tests
    "tests": ["__init__.py"],
}

def ensure_init_files(folder_path: str):
    """Ensure __init__.py exists in each level of nested directories"""
    parts = folder_path.split(os.sep)
    current_path = PROJECT_DIR
    for part in parts:
        current_path = os.path.join(current_path, part)
        os.makedirs(current_path, exist_ok=True)
        init_file = os.path.join(current_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f"# {part} package\n")

def create_folders_and_files():
    """Create the folder structure and initialize files"""
    for folder, files in FOLDER_STRUCTURE.items():
        folder_path = os.path.join(PROJECT_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)

        # Ensure __init__.py at every level
        ensure_init_files(folder)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path) and file.endswith(".py"):
                with open(file_path, "w") as f:
                    content = "# {}\n".format(file)
                    if file == "db.py":
                        content += """
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://awasekhirnisyed:postgres@localhost:5432/profxdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
                        """.strip()
                    elif file == "config.py":
                        content += """
# Configuration values
class Config:
    DEBUG = True
    SECRET_KEY = "your-secret-key-here"
                        """.strip()
                    elif file == "dependencyinjection.py":
                        content += """
# Dependency injection setup will go here
                        """.strip()
                    elif file == "logger.py":
                        content += """
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
                        """.strip()
                    elif file == "exceptions.py":
                        content += """
class UnauthorizedException(Exception):
    pass

class BadRequestException(Exception):
    pass

class NotFoundException(Exception):
    pass
                        """.strip()
                    f.write(content)

def create_main_app():
    """Create main app file"""
    main_path = os.path.join(PROJECT_DIR, "mainapp.py")
    if not os.path.exists(main_path):
        with open(main_path, "w") as f:
            f.write("""from flask import Flask
from flask_restx import Api
from adapters.http.controllers.example_controller import example_ns

app = Flask(__name__)
api = Api(app)
api.add_namespace(example_ns)

if __name__ == '__main__':
    app.run(debug=True)
""")

def create_env_file():
    """Create .env file"""
    env_path = os.path.join(PROJECT_DIR, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("FLASK_APP=mainapp.py\nFLASK_ENV=development\nSECRET_KEY=your-secret-key-here")

def main():
    print("Creating Hexagonal Architecture Project Structure...")
    create_folders_and_files()
    create_main_app()
    create_env_file()
    print("✅ Project structure created successfully!")

if __name__ == "__main__":
    main()
