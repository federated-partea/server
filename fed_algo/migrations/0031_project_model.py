# Generated by Django 3.2.5 on 2021-09-02 15:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0030_alter_project_privacy_level'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='model',
            field=models.FileField(null=True, upload_to=''),
        ),
    ]
