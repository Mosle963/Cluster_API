# Generated by Django 5.1.1 on 2024-10-02 19:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("myapp", "0002_cluster_records_delete_globalsmodel"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="cluster_records",
            name="from_date",
        ),
    ]
