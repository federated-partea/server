# Generated by Django 3.0.4 on 2020-11-04 13:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0005_auto_20201029_1030'),
    ]

    operations = [
        migrations.AlterField(
            model_name='projecttoken',
            name='last_action',
            field=models.CharField(default='Unused', max_length=20),
        ),
    ]
