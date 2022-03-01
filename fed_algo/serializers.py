from rest_framework import serializers

from fed_algo.models import AppUser, Project, ProjectToken, TrafficLog
from fed_algo.storage import dump_blob
from fed_algo.transformations import serialize_bytes


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    is_staff = serializers.CharField(read_only=True)

    def create(self, validated_data):
        user = super(UserSerializer, self).create(validated_data)
        user.set_password(validated_data['password'])
        user.save()
        return user

    class Meta:
        model = AppUser
        fields = ('id', 'username', 'email', 'is_staff', 'password',)
        write_only_fields = ('password',)
        read_only_fields = ('id', 'is_staff', 'is_superuser', 'is_active', 'date_joined',)


class LocalProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = (
            'id', 'state', 'name', 'method', 'privacy_level', 'conditions', 'penalizer', 'l1_ratio', 'timeline',
            'master_token', 'owner_id', 'created_at', 'sample_number', 'c_index', 'error_message', 'smpc', 'from_time',
            'to_time', 'step_size', 'max_iters', 'number_of_sites')
        read_only_fields = (
            'id', 'state', 'name', 'method', 'privacy_level', 'conditions', 'timeline', 'master_token', 'owner_id',
            'created_at')


class ProjectSerializer(serializers.ModelSerializer):
    roles = serializers.SerializerMethodField()
    token = serializers.SerializerMethodField()

    def create(self, validated_data):
        validated_data['owner_id'] = self.context['request'].user.id
        proj = super(ProjectSerializer, self).create(validated_data)
        dump_blob(f'p{proj.id}_aggregate', serialize_bytes(None))
        dump_blob(f'p{proj.id}_data', serialize_bytes(None))
        return proj

    def get_roles(self, instance):
        roles = []

        if instance.owner == self.context['request'].user:
            roles.append('owner')

        if self.context['request'].user in AppUser.objects.filter(projects__project=instance).all():
            roles.append('contributor')

        return roles

    def get_token(self, instance):
        try:
            token = ProjectToken.objects.get(project=instance, user=self.context['request'].user)
            return token.token
        except ProjectToken.DoesNotExist:
            return None

    class Meta:
        model = Project
        fields = (
            'id', 'state', 'step', 'method', 'privacy_level', 'conditions', 'penalizer', 'l1_ratio', 'timeline',
            'master_token', 'token', 'name', 'sample_number', 'from_time', 'to_time', 'step_size', 'max_iters',
            'owner_id', 'c_index', 'error_message', 'smpc', 'number_of_sites',
            'created_at', 'roles', 'run_start', 'run_end')
        read_only_fields = (
            'id', 'step', 'token', 'owner_id', 'created_at', 'run_start', 'run_end', 'plot')


class TokenSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectToken
        fields = ('id', 'token', 'project_id', 'user_id', 'name', 'last_action', 'created_at',)


class TrafficLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrafficLog
        fields = ('id', 'direction', 'token_id', 'state', 'step', 'size', 'preview', 'created_at',)
