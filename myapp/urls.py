from django.urls import path
from .views import PredictView, PreprocessView, RetrainView, ApplyModelView, DeleteModelView,ListCluster_records

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("preprocess/", PreprocessView.as_view(), name="preprocess"),
    path("retrain/", RetrainView.as_view(), name="retrain"),
    path("apply/",ApplyModelView.as_view(),name ="apply"),
    path("delete/",DeleteModelView.as_view(),name = "delete"),
    path("list/", ListCluster_records.as_view(), name="cluster_list"),
]
