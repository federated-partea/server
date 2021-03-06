# Generated by Django 3.0.4 on 2021-05-20 17:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0021_auto_20210520_0751'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='conditions',
            field=models.CharField(default='', max_length=255),
        ),
        migrations.AlterField(
            model_name='resultrow',
            name='values',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='resulttable',
            name='columns',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='resulttable',
            name='plot',
            field=models.TextField(null=True),
        ),
    ]
