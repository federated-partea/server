FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG SECRET_KEY
ENV SECRET_KEY=$SECRET_KEY

WORKDIR /usr/src/fed_algo/

COPY ./requirements.txt /usr/src/fed_algo/requirements.txt
RUN pip3 install --upgrade pip \
    && pip3 install -r /usr/src/fed_algo/requirements.txt \
    && pip3 install gunicorn \
    && rm -rf /root/.cache/pip
    
COPY docker-entrypoint.sh /entrypoint.sh

COPY . /usr/src/fed_algo/

RUN mkdir data

EXPOSE 8000

ENTRYPOINT ["sh", "/entrypoint.sh"]
