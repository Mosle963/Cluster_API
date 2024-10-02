# Generated by Django 5.1.1 on 2024-10-02 19:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("myapp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Cluster_records",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("addedDate", models.DateField(auto_now_add=True)),
                ("silhouette_score", models.FloatField()),
                ("calinski_harabasz_score", models.FloatField()),
                ("number_of_clusters", models.IntegerField()),
                ("total_records", models.IntegerField()),
                ("word2vec_vector_size", models.IntegerField()),
                ("word2vec_window_size", models.IntegerField()),
                ("word2vec_word_min_count_percentage", models.FloatField()),
                ("from_date", models.DateField()),
                ("applied", models.BooleanField()),
                ("inertia", models.FloatField()),
                ("test_id", models.IntegerField()),
                ("timestamp", models.CharField(max_length=20)),
            ],
            options={
                "ordering": ["-id"],
            },
        ),
        migrations.DeleteModel(
            name="GlobalsModel",
        ),
    ]
