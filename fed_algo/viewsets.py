import csv
import logging
import random
import string
import threading
from io import StringIO

from django.db.models import Q
from django.http import HttpResponseForbidden, HttpResponse
from django.utils import timezone
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from fed_algo.models import AppUser, Project, ProjectToken, TrafficLog
from fed_algo.serializers import UserSerializer, ProjectSerializer, TokenSerializer, TrafficLogSerializer
from fed_algo.server_funcs import initialize
from fed_algo.transformations import unescape_cell, CELL_SEPARATOR

logger = logging.getLogger(__name__)


class UserViewSet(viewsets.ModelViewSet):
    queryset = AppUser.objects.all()
    serializer_class = UserSerializer


class ProjectViewSet(viewsets.ModelViewSet):
    serializer_class = ProjectSerializer

    def update(self, request, *args, **kwargs):
        logger.debug("Update project information")

        data = request.data
        project_id = data["_id"]
        logger.debug("Update project" + str(project_id))
        p = Project.objects.get(id=project_id)
        p.method = data["_method"]
        p.timeline = data["_timeline"]
        p.state = data["_state"]
        p.smpc = data["_smpc"]
        if p.smpc:
            p.from_time = data["_from_time"]
            p.to_time = data["_to_time"]
            p.step_size = data["_step_size"]
        if p.method == "univariate":
            p.privacy_level = data["_privacy_level"]
            p.conditions = data["_conditions"]
        if p.method == "cph":
            p.penalizer = data["_penalizer"]
            p.l1_ratio = data["_l1_ratio"]
            p.max_iters = data["_max_iters"]
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
        master_token = ''.join(random.choice(chars) for _ in range(32))
        p.master_token = master_token
        p.save()

        return Response(True)

    def get_queryset(self):
        user = self.request.user
        logger.debug(f'[{user}] Get project')
        return Project.objects.filter(Q(owner=user) | Q(members__user=user)).distinct()

    def destroy(self, request, *args, **kwargs):
        proj = self.get_object()
        logger.debug(f'[{proj.id}] Destroy project')

        if self.get_object().owner != self.request.user:
            return HttpResponseForbidden('Only owners can delete a project.')

        return super(ProjectViewSet, self).destroy(request)

    @action(detail=True, methods=['get'])
    def tokens(self, request, *args, **kwargs):
        proj = self.get_object()
        logger.debug(f'[{proj.id}] Show tokens')

        if proj.owner != request.user:
            logger.warning("Tokens can only be viewed by the project owner")
            return HttpResponseForbidden('Tokens can only be viewed by the project owner.')

        return Response(TokenSerializer(many=True).to_representation(proj.members))

    @action(detail=True, methods=['get'])
    def traffic(self, request, *args, **kwargs):
        proj = self.get_object()
        logger.debug(f'[{proj.id}] Get project traffic')
        if proj.owner != request.user:
            logger.warning("Traffic can only be viewed by the project owner")
            return HttpResponseForbidden('Traffic can only be viewed by the project owner.')

        logs = TrafficLog.objects.filter(token__in=proj.members.all()).order_by('-created_at')
        return Response(TrafficLogSerializer(many=True).to_representation(logs))

    @action(detail=True, methods=['post'])
    def create_token(self, request, *args, **kwargs):
        proj = self.get_object()
        logger.debug(f'[{proj.id}] Create token')
        if proj.owner != request.user:
            logger.warning("Tokens can only be created by the project owner")
            return HttpResponseForbidden('Tokens can only be created by the project owner.')

        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
        token_str = ''.join(random.choice(chars) for _ in range(32))
        token = ProjectToken.objects.create(token=token_str, project=proj, user=None)

        return Response(TokenSerializer().to_representation(token))

    @action(detail=True, methods=['post'])
    def run_project(self, request, *args, **kwargs):
        proj = self.get_object()
        logger.info(f'Run project {proj.id}')

        if proj.owner != request.user:
            logger.warning("Project can only be started by the project owner")
            return HttpResponseForbidden('Project can only be started by the project owner.')
        token_count = ProjectToken.objects.filter(project=proj).count()
        unused_token_count = ProjectToken.objects.filter(project=proj, user__isnull=True).count()
        if unused_token_count == 0 and token_count > 0:
            if token_count == 1:
                proj.smpc = False
            proj.state = 'running'
            proj.run_start = timezone.now()
            proj.save()

            conn_thr = threading.Thread(target=initialize, args=(proj,))
            conn_thr.start()
            return Response(True)
        else:
            return Response(False)

    @action(detail=True)
    def result_tables(self, request, *args, **kwargs):
        proj = self.get_object()
        logger.debug(f'Get result table {proj.id}')

        return Response([
            {
                'id': table.id,
                'name': table.name,
                'columns': [unescape_cell(v) for v in table.columns.split(CELL_SEPARATOR)],
                'plot': table.plot,
                'row_count': table.rows.count(),
            } for table in proj.result_tables.all()])

    @action(detail=True)
    def result_table(self, request, *args, **kwargs):
        proj = self.get_object()
        table_id = int(request.GET.get('tid'))

        table = proj.result_tables.get(id=table_id)
        # TODO: Allow specifying offsets
        return Response([
            {
                'id': row.id,
                'columns': [unescape_cell(v) for v in row.values.split(CELL_SEPARATOR)]
            } for row in table.rows.order_by('created_at')])

    @action(detail=True)
    def result_table_csv(self, request, *args, **kwargs):
        proj = self.get_object()
        table_id = int(request.GET.get('tid'))

        table = proj.result_tables.get(id=table_id)

        fieldnames = [unescape_cell(v) for v in table.columns.split(CELL_SEPARATOR)]

        csv_data = []
        for row in table.rows.order_by('created_at').all():
            row_dict = {
                'id': row.index,
            }
            for i, v in enumerate(row.values.split(CELL_SEPARATOR)):
                row_dict[fieldnames[i]] = unescape_cell(v)
            csv_data.append(row_dict)

        csv_buf = StringIO()

        writer = csv.DictWriter(csv_buf, fieldnames=['id'] + fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

        proj_name = str(table.project.name).lower()
        proj_name = proj_name.replace(' ', '_')

        response = HttpResponse(csv_buf.getvalue(), content_type='text/csv')
        participants = proj.members.all().count()
        analysis = proj.method
        smpc = ""
        if proj.smpc:
            smpc = "_smpc"
        response[
            'Content-Disposition'] = f'attachment; filename="{proj_name}_{participants}_{analysis}{smpc}.csv"'
        return response


class TokenViewSet(viewsets.ModelViewSet):
    serializer_class = TokenSerializer

    def get_queryset(self):
        return ProjectToken.objects.filter(project__owner=self.request.user)
