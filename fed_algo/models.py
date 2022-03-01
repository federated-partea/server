from django.contrib.auth.models import AbstractUser
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from fed_algo.transformations import CELL_SEPARATOR, escape_cell

PROJECT_STATES = (
    ('pre_start', 'Pre start'), ('initialized', 'initialized'), ('waiting', 'Running'), ('running', 'Running'),
    ('finished', 'Finished'), ('error', 'Error'))
METHODS = (
    ('unviariate', 'Univariate analysis'), ('cox', 'Cox proportional hazards model'))
TIMELINE = (('years', 'Years'), ('months', 'Months'), ('weeks', 'Weeks'), ('days', 'Days'), ('none', 'Complete'))


class AppUser(AbstractUser):
    pass


class Project(models.Model):
    owner = models.ForeignKey('AppUser', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    state = models.CharField(max_length=31, choices=PROJECT_STATES, default='pre_start')
    method = models.CharField(max_length=255, choices=METHODS, default='univariate')
    privacy_level = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(20)], default=0)
    number_of_sites = models.IntegerField(default=0)
    smpc = models.BooleanField(default=False)
    from_time = models.FloatField(default=0)
    to_time = models.FloatField(default=100)
    step_size = models.FloatField(validators=[MinValueValidator(0.0)], default=1.0)
    conditions = models.CharField(max_length=255, default='')
    penalizer = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=0.0)
    l1_ratio = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=0.0)
    max_iters = models.PositiveIntegerField(default=10)
    timeline = models.CharField(max_length=10, choices=TIMELINE, default='years')
    master_token = models.CharField(max_length=255, default='')
    run_start = models.DateTimeField(null=True)
    run_end = models.DateTimeField(null=True)
    sample_number = models.PositiveIntegerField(default=0)
    c_index = models.CharField(null=True, max_length=20)
    error_message = models.CharField(null=True, max_length=255)

    internal_state = models.CharField(default="init", max_length=50)
    step = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def create_result_table(self, table_name, columns, plot):
        ResultTable.objects.create(project=self,
                                   name=table_name,
                                   columns=CELL_SEPARATOR.join([escape_cell(v) for v in columns]),
                                   plot=plot)

    def create_result_row(self, table_name, row_name, values):
        table = ResultTable.objects.get(project=self, name=table_name)
        table.rows.create(index=row_name,
                          values=CELL_SEPARATOR.join([escape_cell(v) for v in values]))

    def create_dataframe(self, table_name, b):
        table = ResultTable.objects.get(project=self, name=table_name)
        Dataframe.objects.create(table=table, file=b)


class ProjectToken(models.Model):
    token = models.CharField(max_length=255)
    project = models.ForeignKey('Project', on_delete=models.CASCADE, related_name='members')
    user = models.ForeignKey('AppUser', on_delete=models.CASCADE, null=True, blank=True, related_name='projects')
    name = models.CharField(max_length=255, null=True, blank=True, default='')
    last_action = models.CharField(max_length=20, default='Unused')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('project', 'user',)


class ProjectLog(models.Model):
    message = models.CharField(max_length=255)
    project = models.ForeignKey('Project', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)


class TrafficLog(models.Model):
    token = models.ForeignKey('ProjectToken', on_delete=models.CASCADE, related_name='traffic_logs')
    direction = models.CharField(max_length=255, choices=(('in', 'incoming'), ('out', 'outgoing'),))
    state = models.CharField(max_length=255, choices=PROJECT_STATES)
    step = models.PositiveIntegerField()
    size = models.PositiveIntegerField(null=True, blank=True)
    preview = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)


class ResultTable(models.Model):
    name = models.CharField(max_length=255)
    columns = models.CharField(max_length=255)
    project = models.ForeignKey('Project', on_delete=models.CASCADE, related_name='result_tables')
    plot = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ResultRow(models.Model):
    table_name = models.CharField(max_length=255)
    index = models.CharField(max_length=255)
    values = models.CharField(max_length=255)
    project = models.ForeignKey('ResultTable', on_delete=models.CASCADE, related_name='rows')
    created_at = models.DateTimeField(auto_now_add=True)


class Dataframe(models.Model):
    table = models.ForeignKey('ResultTable', on_delete=models.CASCADE, related_name='dataframe')
    file = models.BinaryField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
