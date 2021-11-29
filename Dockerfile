FROM python:3.9
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
# enter entry point parameters executing the container
ENTRYPOINT ["python", "./runserver.py"]
# exposing the port to match the port in the runserver.py file
EXPOSE 5555