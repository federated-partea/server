# Generated by Django 3.0.4 on 2021-05-17 11:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0015_auto_20210517_1112'),
    ]

    operations = [
        migrations.AddField(
            model_name='resulttable',
            name='plot',
            field=models.CharField(max_length=2550, null=True),
        ),
    ]
