FROM gcr.io/deeplearning-platform-release/tf-cpu.2-6

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 4 --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app

