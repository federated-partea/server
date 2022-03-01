import os

import redis

from fed_algo.models import Project
from fed_algo.settings import STORAGE_PATH, STORAGE_MODE

r = None

if STORAGE_MODE == 'redis':
    r = redis.Redis(host=os.environ.get('REDIS_HOST', 'localhost'), port=os.environ.get('REDIS_PORT', 6379), db=0)


def dump_blob(name, data):
    if STORAGE_MODE == 'file':
        f = open(f'{STORAGE_PATH}/{name}.pkl', 'wb')
        f.write(data)
        f.close()
    elif STORAGE_MODE == 'redis':
        r.set(name, data)


def load_blob(name):
    if STORAGE_MODE == 'file':
        f = open(f'{STORAGE_PATH}/{name}.pkl', 'rb')
        data = f.read()
        f.close()

        return data
    elif STORAGE_MODE == 'redis':
        return r.get(name)


def delete_blob(name):
    if STORAGE_MODE == 'file':
        try:
            os.remove(f'{STORAGE_PATH}/{name}.pkl')
        except FileNotFoundError:
            pass
    elif STORAGE_MODE == 'redis':
        r.delete(name)


def exists_blob(name):
    if STORAGE_MODE == 'file':
        return os.path.exists(f'{STORAGE_PATH}/{name}.pkl')
    elif STORAGE_MODE == 'redis':
        return r.exists(name) > 0


def cleanup(proj: Project):
    try:
        os.remove(f'{STORAGE_PATH}/{proj.id}.lock')
    except Exception as e:
        print(e)
    try:
        os.remove(f'{STORAGE_PATH}/p{proj.id}_aggregate.pkl')
    except Exception as e:
        print(e)
    try:
        os.remove(f'{STORAGE_PATH}/p{proj.id}_data.pkl')
    except Exception as e:
        print(e)
    try:
        os.remove(f'{STORAGE_PATH}/p{proj.id}_memory.pkl')
    except Exception as e:
        print(e)
    for mem in proj.members.all():
        print(mem.id)
        try:
            os.remove(f'{STORAGE_PATH}/p{proj.id}_{mem.id}.pkl')
        except Exception as e:
            print(e)
        try:
            os.remove(f'{STORAGE_PATH}/c{mem.id}_data.pkl')
        except Exception as e:
            print(e)
