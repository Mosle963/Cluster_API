import numpy as np
import gensim
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from datetime import datetime




def gensim_fun(word2vec_window_size,
               word2vec_word_min_count_percentage,
               word2vec_vector_size,
               text_data,timestamp):

    #Calculate the minimum document frequency for a word to be considered based on the percentage passes
    word2vec_word_min_count = int(len(text_data) * word2vec_word_min_count_percentage)

    #initilize the model
    word2vecmodel = gensim.models.Word2Vec(
          window = word2vec_window_size,
          min_count = word2vec_word_min_count,
          vector_size = word2vec_vector_size)

    #vectorize docs using gensim preprocessing
    corpus_iterable =[]
    for text in text_data:
        vector = gensim.utils.simple_preprocess(text)
        corpus_iterable.append(vector)

    #build vocabulary and train word2vec model
    word2vecmodel.build_vocab(corpus_iterable)
    word2vecmodel.train(corpus_iterable,
                        total_examples=word2vecmodel.corpus_count,
                        epochs = word2vecmodel.epochs)
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    word2vecmodel_path = Path(BASE_DIR, f'word2vecmodel_{timestamp}.joblib')

    joblib.dump(word2vecmodel, word2vecmodel_path)
    #replace each doc with a vector calculated as mean of all words vectors in the doc
    vectors=[]
    for text in corpus_iterable:
        vectors.append(word2vecmodel.wv.get_mean_vector(text))

    #change the diminsions of the vectors array to be suitable for training functions
    vectors_2d = np.stack(vectors)

    return vectors_2d





def kmeans_fun(n_clusters,max_iter,n_init,vectors,timestamp):

    model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=max_iter,n_init=n_init)
    labels = model.fit_predict(vectors)


    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    kmeans_path = Path(BASE_DIR, f'Kmeans_model_{timestamp}.joblib')
    joblib.dump(model, kmeans_path)


    cluster_labels = model.labels_
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    label_count = {}

    for label, count in zip(unique_labels, counts):
      label_count[str(label)] = count

    return {'labels':labels,'label_count':label_count,'inertia':round(model.inertia_,6)}



def sil_fun(vectors,labels):
    try:
        sil_score =float(silhouette_score(vectors, labels))
        sil_score = round(sil_score,3)
    except Exception as e:
        print(f"An error occurred: {e}")
        sil_score = 0
    return sil_score


def ch_fun(vectors,labels):
    try:
        ch_score = calinski_harabasz_score(vectors, labels)
        ch_score = round(ch_score,3)
    except Exception as e:
        print(f"An error occurred: {e}")
        ch_score = 0
    return ch_score


def gensim_kmeans_fun(text_data,
                      word2vec_window_size,
                      word2vec_word_min_count_percentage,
                      word2vec_vector_size,
                      n_clusters,
                      max_iter,
                      n_init,
                      ):

    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    vectors = gensim_fun( word2vec_window_size,word2vec_word_min_count_percentage,
                          word2vec_vector_size,text_data,timestamp)
    kmeans_results = kmeans_fun(n_clusters,max_iter,n_init,vectors,timestamp)
    return {
                'label_count':kmeans_results['label_count'],
                'ch_score' : ch_fun(vectors,kmeans_results['labels']),
                'sil_score' : sil_fun(vectors,kmeans_results['labels']),
                'inertia' : kmeans_results['inertia'],
                'timestamp' : timestamp,
                'word2vec_word_min_count_percentage' : word2vec_word_min_count_percentage,
                'word2vec_vector_size' : word2vec_vector_size,
                'word2vec_window_size' : word2vec_window_size,
                'n_clusters' : n_clusters
            }




def test_number_of_clusters_gensim_kmeans(data):
    text_data = data ['textList']
    start_number = data ['start_number']
    end_number = data ['end_number']
    step = data ['step']
    word2vec_word_min_count_percentage = data ['word2vec_word_min_count_percentage']
    word2vec_vector_size = data ['word2vec_vector_size']
    word2vec_window_size = data ['word2vec_window_size']


    results = []

    if word2vec_word_min_count_percentage<0.01:
        word2vec_word_min_count_percentage = 0.35

    for n_clusters in range(start_number, end_number + 1, step):
        result = gensim_kmeans_fun(text_data=text_data,
                      word2vec_window_size=word2vec_window_size,
                      word2vec_word_min_count_percentage=word2vec_word_min_count_percentage,
                      word2vec_vector_size=word2vec_vector_size,
                      n_clusters=n_clusters,
                      max_iter=5000,
                      n_init=10)
        results.append(result)

    return results