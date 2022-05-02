# Generated by Django 4.0 on 2022-02-10 18:08

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0038_remove_project_n_estimators'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='privacy_level',
            field=models.FloatField(default=0, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(20)]),
        ),
    ]