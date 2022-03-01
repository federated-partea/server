import logging
import random
import string
import threading

from django.contrib.auth import authenticate
from django.http import HttpResponseBadRequest, HttpResponse, HttpResponseNotFound, HttpResponseForbidden
from django.utils.encoding import smart_str
from filelock import FileLock
from rest_framework import generics
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from fed_algo.models import AppUser, ProjectToken, TrafficLog, Project
from fed_algo.serializers import UserSerializer, LocalProjectSerializer
from fed_algo.server_funcs import aggregate
from fed_algo.settings import STORAGE_PATH
from fed_algo.storage import load_blob, dump_blob, exists_blob
from fed_algo.transformations import deserialize_bytes

logger = logging.getLogger(__name__)


class UserInfo(APIView):

    def get(self, request):
        return Response(UserSerializer().to_representation(request.user))


class Logs(APIView):

    def get(self, request):
        response = HttpResponse(
            mimetype='application/force-download')
        response['Content-Disposition'] = 'attachment; filename=%s' % smart_str("partea_debug.log")
        response['X-Sendfile'] = smart_str("/tmp/partea_debug.log")

        return response


class TokenBlacklistView(APIView):

    def post(self, request):
        token = RefreshToken(request.data.get('refresh'))
        token.blacklist()
        return Response({'success': True})


class UserCreateView(generics.CreateAPIView):
    queryset = AppUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = (AllowAny,)


class ClientProjectView(APIView):
    permission_classes = (AllowAny,)

    def get(self, request):
        token = request.GET.get('token')
        token_ins = ProjectToken.objects.get(token=token)
        return Response(LocalProjectSerializer().to_representation(token_ins.project))

    def post(self, request):
        token = request.data.get('token')
        username = request.data.get('username')
        password = request.data.get('password')
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
        try:
            proj = Project.objects.get(master_token=token)
            logger.debug("Tokens is a master token")
            if proj.state == 'finished' or proj.state == 'running':
                logger.warning("Project has already finished")
                return HttpResponseNotFound("This project has already finished. ")
            else:
                logger.debug("Create personal token from master token")
                token = ''.join(random.choice(chars) for _ in range(32))
                proj_token = ProjectToken.objects.create(token=token, project=proj)
                proj_token.save()
                tmp_pw = ''.join(random.choice(chars) for _ in range(10))
                tmp_username = ''.join(random.choice(chars) for _ in range(10))
                user = AppUser.objects.create(username=tmp_username, password=tmp_pw)
                user.save()
        except Project.DoesNotExist:
            logger.debug("Authenticate user...")
            user = authenticate(username=username, password=password)
        try:
            logger.debug("Check token...")
            token_ins = ProjectToken.objects.get(token=token)
        except ProjectToken.DoesNotExist:
            logger.warning("Token does not exist")
            return HttpResponseNotFound("Token does not exist. ")

        proj = token_ins.project

        if token_ins.user:
            logger.warning("Token already used...")
            return HttpResponseForbidden('Token already used')

        if ProjectToken.objects.filter(user=user, project=proj).exists():
            logger.warning("User has already joined with a different token")

            return HttpResponseForbidden('User has already joined with a different token')

        token_ins.user = user
        token_ins.name = username
        token_ins.last_action = "Joined"
        token_ins.save()
        proj.number_of_sites += 1
        proj.save()
        client_id = int(ProjectToken.objects.get(token=token).id)

        return Response({'project': LocalProjectSerializer().to_representation(proj), 'token': token_ins.token,
                         'client_id': client_id})


class ClientTaskView(APIView):
    permission_classes = (AllowAny,)

    def get(self, request):
        mode = request.GET.get('mode')
        token = request.GET.get('token')
        client = request.GET.get('client')
        token_ins = ProjectToken.objects.get(token=token)
        token_ins.last_action = "Running"
        proj = token_ins.project
        token_ins.save()
        if mode == 'data':
            if "smpc" not in token_ins.project.internal_state:
                agg_bytes = load_blob(f'p{token_ins.project_id}_aggregate')
            else:
                agg_bytes = load_blob(f'p{token_ins.project_id}_{client}')
            TrafficLog.objects.create(token=token_ins,
                                      direction='out',
                                      state=token_ins.project.internal_state,
                                      step=token_ins.project.step,
                                      size=len(agg_bytes),
                                      preview=str(deserialize_bytes(agg_bytes))[:50])
            return HttpResponse(agg_bytes)

        elif mode == 'state':
            resp = {
                'state': token_ins.project.state,
                'internal_state': token_ins.project.internal_state,
                'step': token_ins.project.step,
                'number_of_participants': proj.number_of_sites
            }

            return Response(resp)

    def post(self, request):
        token = request.GET.get('token')
        token_ins = ProjectToken.objects.get(token=token)
        data = request.body

        if not data:
            return HttpResponseBadRequest("Data must not be null.")

        dump_blob(f'c{token_ins.id}_data', data)

        TrafficLog.objects.create(token=token_ins,
                                  direction='in',
                                  state=token_ins.project.internal_state,
                                  step=token_ins.project.step,
                                  size=len(data),
                                  preview=str(deserialize_bytes(data))[:50])

        proj = token_ins.project
        lock = FileLock(f'{STORAGE_PATH}/{proj.id}.lock')
        if all((exists_blob(f'c{mem.id}_data') for mem in proj.members.all())):
            threading.Thread(target=aggregate, args=(proj, lock)).start()

        return Response('')
