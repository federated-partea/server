# Generated by Django 3.2.5 on 2021-07-21 09:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0027_auto_20210720_1646'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='oob',
            field=models.CharField(max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='test_sample_number',
            field=models.PositiveIntegerField(default=0),
        ),
    ]