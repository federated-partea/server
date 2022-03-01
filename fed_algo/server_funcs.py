import logging

from filelock import FileLock

from fed_algo.models import Project
from fed_algo.server_app import ServerApp
from fed_algo.storage import delete_blob, load_blob, dump_blob, cleanup
from fed_algo.transformations import deserialize_bytes, serialize_bytes

logger = logging.getLogger(__name__)


def initialize(project):
    def ready_func(ready, agg=None, data=None, memory=None):
        dump_blob(f'p{project.id}_aggregate', serialize_bytes(agg))
        dump_blob(f'p{project.id}_data', serialize_bytes(data))
        dump_blob(f'p{project.id}_memory', serialize_bytes(memory))

        project.state = 'waiting' if not ready else 'finished'
        project.step += 1
        project.save()

    logger.info(f'[{project.id}]Initialize server for aggregation')
    ServerApp(project=project, ready_func=ready_func, old_aggregate=None, old_data=None, memory=None).initialize()


def aggregate(proj: Project, lock: FileLock):
    with lock:
        logger.info(f'[{proj.id}] Aggregate local client data')
        client_data = {}

        for mem in proj.members.all():
            logger.debug(f'[{proj.id}] Add local client data to aggregation: {str(mem.id)}')
            client_bytes = load_blob(f'c{mem.id}_data')
            deserialized = deserialize_bytes(client_bytes)
            client_data[mem.id] = deserialized
            delete_blob(f'c{mem.id}_data')

        def ready_func(ready, agg=None, data=None, memory=None):
            dump_blob(f'p{proj.id}_aggregate', serialize_bytes(agg))
            if "smpc" in proj.internal_state:
                for mem in proj.members.all():
                    dump_blob(f'p{proj.id}_{mem.id}', serialize_bytes(agg[mem.id]))
            dump_blob(f'p{proj.id}_data', serialize_bytes(data))
            dump_blob(f'p{proj.id}_memory', serialize_bytes(memory))

            if not ready:
                proj.state = 'waiting'
            else:
                cleanup(proj)

            proj.step += 1
            proj.save()

        proj.state = 'running'
        proj.save()

        proj_aggregate = deserialize_bytes(load_blob(f'p{proj.id}_aggregate'))
        proj_data = deserialize_bytes(load_blob(f'p{proj.id}_data'))
        proj_memory = deserialize_bytes(load_blob(f'p{proj.id}_memory'))

        ServerApp(proj, ready_func, proj_aggregate, proj_data, proj_memory).aggregate(client_data)
