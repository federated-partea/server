# Generated by Django 3.2.5 on 2021-11-25 10:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0033_auto_20211125_1047'),
    ]

    operations = [
        migrations.RenameField(
            model_name='project',
            old_name='min_time',
            new_name='from_time',
        ),
        migrations.RenameField(
            model_name='project',
            old_name='max_steps',
            new_name='max_iters',
        ),
        migrations.RenameField(
            model_name='project',
            old_name='max_time',
            new_name='to_time',
        ),
    ]