#!/bin/bash

python3 manage.py makemigrations
python3 manage.py migrate
gunicorn fed_algo:application --bind 0.0.0.0:8000 --timeout 1200 --workers 8
