FROM jupyter/scipy-notebook

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

RUN python3 inference.py