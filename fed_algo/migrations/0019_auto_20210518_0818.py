# Generated by Django 3.0.4 on 2021-05-18 08:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0018_auto_20210517_1540'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='sample_number',
            field=models.PositiveIntegerField(default=0),
        ),
    ]
