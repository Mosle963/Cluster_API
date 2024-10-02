import joblib
import gensim
from pathlib import Path
import os
from ..models import Cluster_records


def predict(text):


    Kmeans_file_name = "Kmeans_model_default.joblib"
    Word2vec_file_name = "word2vecmodel_default.joblib"
    try:
        applied_record = Cluster_records.objects.get(applied=True)
        timestamp = applied_record.timestamp
        Kmeans_file_name = f"Kmeans_model_{timestamp}.joblib"
        Word2vec_file_name = f"word2vecmodel_{timestamp}.joblib"
    except:
        pass

    BASE_DIR = Path(__file__).resolve().parent
    kmeans_path = Path(BASE_DIR, Kmeans_file_name)
    print(kmeans_path)
    word2vecmodel_path = Path(BASE_DIR, Word2vec_file_name)
    model = joblib.load(kmeans_path)
    word2vecmodel = joblib.load(word2vecmodel_path)
    try:
        vector = gensim.utils.simple_preprocess(text)
        w2v = word2vecmodel.wv.get_mean_vector(vector)
        vector_2d = w2v.reshape(1, -1)
        label = model.predict(vector_2d)
        return label
    except:
        return [-1]
