# Generated by Django 3.0.4 on 2020-10-29 10:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0003_auto_20201023_1424'),
    ]

    operations = [
        migrations.AddField(
            model_name='projecttoken',
            name='lastAction',
            field=models.CharField(default='Joined', max_length=20),
        ),
    ]
