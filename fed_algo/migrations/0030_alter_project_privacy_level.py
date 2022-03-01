# Generated by Django 3.2.5 on 2021-07-31 18:19

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0029_alter_project_max_depth'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='privacy_level',
            field=models.IntegerField(default=0, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(20)]),
        ),
    ]
