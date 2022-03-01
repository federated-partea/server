"""fed_algo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import include, re_path
from django.contrib import admin
from rest_framework import routers
from rest_framework_simplejwt.views import TokenVerifyView, TokenRefreshView, TokenObtainPairView

from fed_algo.models import ProjectToken, Project
from fed_algo.views import UserCreateView, TokenBlacklistView, UserInfo, ClientProjectView, ClientTaskView, Logs
from fed_algo.viewsets import UserViewSet, ProjectViewSet, TokenViewSet

router = routers.DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'projects', ProjectViewSet, basename=Project)
router.register(r'tokens', TokenViewSet, basename=ProjectToken)

urlpatterns = [
    re_path(r'^', include(router.urls)),

    re_path(r'^user/info/', UserInfo.as_view()),

    re_path(r'^logs/', Logs.as_view()),

    re_path(r'^client/project/', ClientProjectView.as_view()),
    re_path(r'^client/task/', ClientTaskView.as_view()),
    re_path(r'^auth/signup/$', UserCreateView.as_view()),
    re_path(r'^auth/login/$', TokenObtainPairView.as_view()),
    re_path(r'^auth/token/refresh/$', TokenRefreshView.as_view()),
    re_path(r'^auth/token/verify/$', TokenVerifyView.as_view()),
    re_path(r'^auth/logout/$', TokenBlacklistView.as_view()),

    re_path('admin/', admin.site.urls),
]
