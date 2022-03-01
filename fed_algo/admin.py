from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from fed_algo.models import AppUser, Project, ProjectToken, ResultTable, ResultRow

admin.site.register(AppUser, UserAdmin)

admin.site.register(Project)
admin.site.register(ProjectToken)
admin.site.register(ResultTable)
admin.site.register(ResultRow)
