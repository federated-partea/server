# Generated by Django 3.0.4 on 2021-05-20 07:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0019_auto_20210518_0818'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='errorMessage',
            field=models.CharField(max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='project',
            name='state',
            field=models.CharField(choices=[('pre_start', 'Pre start'), ('initialized', 'initialized'), ('waiting', 'Running'), ('running', 'Running'), ('finished', 'Finished'), ('error', 'Error')], default='pre_start', max_length=31),
        ),
        migrations.AlterField(
            model_name='trafficlog',
            name='state',
            field=models.CharField(choices=[('pre_start', 'Pre start'), ('initialized', 'initialized'), ('waiting', 'Running'), ('running', 'Running'), ('finished', 'Finished'), ('error', 'Error')], max_length=255),
        ),
    ]
