FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Mount your PDFs at /app/data/pdfs and the index at /app/storage
CMD ["uvicorn", "ragbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
