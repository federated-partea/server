# Generated by Django 3.0.4 on 2021-05-17 10:57

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fed_algo', '0012_project_plot'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='project',
            name='plot',
        ),
        migrations.CreateModel(
            name='CoxPlot',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.BinaryField(null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('table', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cox', to='fed_algo.ResultTable')),
            ],
        ),
    ]
