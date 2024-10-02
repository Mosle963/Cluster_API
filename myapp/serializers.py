from rest_framework import serializers
from .models import Cluster_records

class OneentrySerializer(serializers.Serializer):
    text = serializers.CharField()

class RetrainSerializer(serializers.Serializer):
    textList = serializers.ListField(
        child=serializers.CharField()
    )
    word2vec_vector_size = serializers.IntegerField()
    word2vec_window_size = serializers.IntegerField()
    word2vec_word_min_count_percentage = serializers.FloatField()
    start_number = serializers.IntegerField()
    end_number = serializers.IntegerField()
    step = serializers.IntegerField()


class Cluster_records_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Cluster_records
        fields = (
        "id",
        "addedDate",
        "silhouette_score",
        "calinski_harabasz_score",
        "number_of_clusters",
        "total_records",
        "word2vec_vector_size",
        "word2vec_window_size",
        "word2vec_word_min_count_percentage",
        "applied",
        "inertia",
        "test_id",
        "timestamp",)