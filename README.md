# AI Engineering Drawing Evaluation System

Production-oriented web application for evaluating engineering drawing answer sheets against a reference drawing using computer vision, geometric analysis, and batch processing.

## Stack

- Backend: FastAPI, OpenCV, NumPy, SQLAlchemy
- Frontend: React, Vite, TypeScript
- Storage: Local filesystem by default
- Database: SQLite for local development, PostgreSQL-ready via `DATABASE_URL`
- Exports: CSV and Excel

## Folder Structure

```text
backend/
  app/
    api/              FastAPI routes
    core/             configuration
    cv/               preprocessing and feature extraction
    db/               database setup and models
    services/         comparison, feedback, batch jobs, export
    schemas/          pydantic schemas
  main.py
  requirements.txt
frontend/
  src/
    components/       dashboard components
    lib/              API client
    types.ts
  package.json
samples/
  generate_samples.py
storage/
  uploads/
  outputs/
```

## Quick Start

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ..\samples\generate_samples.py
uvicorn main:app --reload --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Main API Endpoints

- `POST /upload-reference`
- `POST /upload-students`
- `POST /process`
- `GET /results`
- `GET /results/{result_id}`
- `GET /jobs/{job_id}`
- `GET /export/csv`
- `GET /export/excel`

## Notes

The local job runner uses FastAPI background tasks for easy development. The job service is isolated so it can be replaced by Celery/RQ in production without changing the API contract.
