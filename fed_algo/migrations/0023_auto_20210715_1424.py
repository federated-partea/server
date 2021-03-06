# Generated by Django 3.0.4 on 2021-07-15 14:24

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0022_auto_20210520_1704'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='l1_ratio',
            field=models.FloatField(default=0.0, validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(1.0)]),
        ),
        migrations.AddField(
            model_name='project',
            name='max_steps',
            field=models.PositiveIntegerField(default=500),
        ),
        migrations.AddField(
            model_name='project',
            name='penalization',
            field=models.FloatField(default=0.0, validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(1.0)]),
        ),
    ]
