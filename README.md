# Math Tutor Backend

Backend for an AI Math Tutor project, built with **FastAPI** and **SymPy**.

## Features
- Step checker that verifies algebra/calculus steps
- Supports equations and expressions
- Simple API ready for frontend integration

## Requirements
- Python 3.10+
- FastAPI, Uvicorn, SymPy, Pydantic, python-dotenv

## Setup
```bash
# clone repo
git clone https://github.com/<your-username>/math-tutor-backend.git
cd math-tutor-backend

# create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Mac/Linux

# install dependencies
pip install -r requirements.txt
