from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import OneentrySerializer,RetrainSerializer,Cluster_records_Serializer
from .algo import ML, predicting,training
from pathlib import Path
from .models import Cluster_records
from rest_framework import generics
import os
class PreprocessView(APIView):
    def post(self, request):
        serializer = OneentrySerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            preprocessed_text = ML.preprocess(data["text"])
            return Response(
                {"preprocessed_text": preprocessed_text}, status=status.HTTP_200_OK
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PredictView(APIView):
    def post(self, request):
        serializer = OneentrySerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            cluster = predicting.predict(data["text"])
            return Response({"cluster": cluster}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)





class RetrainView(APIView):
    def post(self, request):
        serializer = RetrainSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            results = training.test_number_of_clusters_gensim_kmeans(data)
            total_records=len(data['textList'])
            last_test_id = 0
            last_added = Cluster_records.objects.order_by('-id').first()
            if last_added:
                last_test_id = last_added.test_id

            for record in results:
                Cluster_records.objects.create(
                    calinski_harabasz_score = record["ch_score"],
                    silhouette_score = record["sil_score"],
                    number_of_clusters = record["n_clusters"],
                    word2vec_vector_size = record["word2vec_vector_size"],
                    word2vec_window_size = record["word2vec_window_size"],
                    word2vec_word_min_count_percentage = record["word2vec_word_min_count_percentage"],
                    applied = False,
                    inertia = record["inertia"],
                    timestamp = record["timestamp"],
                    test_id = last_test_id + 1,
                    total_records = total_records
                )
            return Response(status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ApplyModelView(APIView):
    def post(self, request):
        serializer = OneentrySerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            record_id = data['text']
            try:
                applied_record = Cluster_records.objects.get(applied=True)
                applied_record.applied=False
                applied_record.save()
            except:
                pass

            if(record_id!='-1'):
                record = Cluster_records.objects.get(id=record_id)
                record.applied=True
                record.save()

            return Response(status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DeleteModelView(APIView):
    def post(self, request):
        serializer = OneentrySerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            record_id = data['text']
            record = Cluster_records.objects.get(id=record_id)
            timestamp = record.timestamp
            record.delete()
            BASE_DIR = Path(__file__).resolve().parent
            kmeans_path = Path(BASE_DIR, f"algo/Kmeans_model_{timestamp}.joblib")
            word2vecmodel_path = Path(BASE_DIR, f"algo/word2vecmodel_{timestamp}.joblib")
            os.remove(kmeans_path)
            os.remove(word2vecmodel_path)
            return Response(status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ListCluster_records(generics.ListAPIView):
    queryset = Cluster_records.objects.all()
    serializer_class = Cluster_records_Serializer