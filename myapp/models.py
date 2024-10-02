from django.db import models


class Cluster_records(models.Model):
    addedDate = models.DateField(auto_now_add=True)
    silhouette_score = models.FloatField()
    calinski_harabasz_score = models.FloatField()
    number_of_clusters = models.IntegerField()
    total_records = models.IntegerField()
    word2vec_vector_size = models.IntegerField()
    word2vec_window_size = models.IntegerField()
    word2vec_word_min_count_percentage = models.FloatField()
    applied = models.BooleanField()
    inertia = models.FloatField()
    test_id = models.IntegerField()
    timestamp = models.CharField(max_length=20)

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return str(self.id)
