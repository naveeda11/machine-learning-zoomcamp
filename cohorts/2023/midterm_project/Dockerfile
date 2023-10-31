FROM python:3.11-slim-bullseye
ENV PYTHONUNBUFFERED True
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

ENV PORT 1234

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 8000

#ENTRYPOINT [ "uvicorn", "predict:app", "--reload", "--host" , "0.0.0.0" ]
CMD exec uvicorn predict:app --host 0.0.0.0 --port ${PORT}
