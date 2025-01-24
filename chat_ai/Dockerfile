FROM python:3.11.4


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


RUN pip install --upgrade pip

WORKDIR /api

COPY --chown=api:api . .
RUN pip install -r requirements.txt