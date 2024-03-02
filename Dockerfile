FROM python:3.8-slim

WORKDIR /opt

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /opt/

COPY app/model /opt/app/model

# Expose the port
EXPOSE 8000

ENV PYTHONUNBUFFERED 1

RUN useradd -m myuser
USER myuser

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
