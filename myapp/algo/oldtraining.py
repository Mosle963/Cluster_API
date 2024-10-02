"""from ..models import Job_Post,cluster_records,Course,Employee,globals

from .predicting import predict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os



from collections import defaultdict

def delete_old_plots():
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder = os.path.join(BASE_DIR, 'static/Images')  
    filename = "sil_score.png" 
    filepath = os.path.join(static_folder, filename)
    print("deleting")
    if os.path.exists(filepath):
        os.remove(filepath)
        print("1")
    filename = "elbow_method.png" 
    filepath = os.path.join(static_folder, filename)

    if os.path.exists(filepath):
        # Delete the existing file
        os.remove(filepath)
        print("2")
    filename = "ch_score.png" 
    filepath = os.path.join(static_folder, filename)
    
    
    if os.path.exists(filepath):
        # Delete the existing file
        os.remove(filepath)
        print("3")
        

def plot_arrays():
    
    first_row = globals.objects.first()
    results = list(cluster_records.objects.filter(applied=False).order_by('-id')[:first_row.number_of_rows])
    results = reversed(results)

    data_dict = defaultdict(list)

    
    for record in results:
        for key, value in record.__dict__.items():
            data_dict[key].append(value)

    array1 = data_dict["silhouette_score"]
    array3 = data_dict["calinski_harabasz_score"]
    n_clusters = data_dict["number_of_clusters"]
    array2 = data_dict["inertia"]

    
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    static_folder = os.path.join(BASE_DIR, 'static/Images')  

    plt.plot(n_clusters, array1, label="Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score")
    plt.xticks(n_clusters)
    plt.vlines(n_clusters, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], colors='gray', linestyles='--', alpha=0.5)

    filename = "sil_score.png" 
    filepath = os.path.join(static_folder, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    plt.savefig(filepath, format='png')  
    plt.clf()



    plt.plot(n_clusters, array2, label="Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("inertia")
    plt.title("Elbow Method")
    plt.xticks(n_clusters)
    plt.vlines(n_clusters, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], colors='gray', linestyles='--', alpha=0.5)
    
    
    filename = "elbow_method.png" 
    filepath = os.path.join(static_folder, filename)
    
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    plt.savefig(filepath, format='png')  
    plt.clf()
    plt.plot(n_clusters, array3, label="Calinski Harabasz Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Calinski Harabasz Score")
    plt.title("Calinski Harabasz Score")
    plt.xticks(n_clusters)
    plt.vlines(n_clusters, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], colors='gray', linestyles='--', alpha=0.5)
   

    
    filename = "ch_score.png" 
    filepath = os.path.join(static_folder, filename)
    
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    plt.savefig(filepath, format='png')  
    plt.clf()
    plt.close()










def train_model(start_date,
                number_of_clusters,
                word2vec_vector_size,
                word2vec_window_size,
                word2vec_word_min_count_percentage):

    first_row = globals.objects.first()
    first_row.training_thread_running = True
    first_row.save()
    
    job_posts = Job_Post.objects.filter(added_date__gte=start_date)
    number_of_records = job_posts.count()
    if word2vec_word_min_count_percentage < 0.01:
        word2vec_word_min_count_percentage = 0.35    

    text_data = list(job_posts.values_list('clusterable_text', flat=True))
   
    result = gensim_kmeans_fun(text_data=text_data,
                      word2vec_window_size=word2vec_window_size,
                      word2vec_word_min_count_percentage=word2vec_word_min_count_percentage,
                      word2vec_vector_size=word2vec_vector_size,
                      n_clusters=number_of_clusters,
                      max_iter=5000,
                      n_init=10,
                      dump=1)
        
    
    for i in range(len(job_posts)):
        job_posts[i].cluster=result['labels'][i]
        job_posts[i].save()

    job_posts = Job_Post.objects.filter(added_date__lt=start_date)
    for i in range(len(job_posts)):
        job_posts[i].cluster = predict(job_posts[i].clusterable_text)
        job_posts[i].save()

    courses = Course.objects.all()
    for course in courses:
        course.cluster = predict(course.clusterable_text)
        course.save()

    employees = Employee.objects.all()
    for emp in employees:
        emp.cluster = predict(emp.clusterable_text)
        emp.save()
    
    cluster_records.objects.create(calinski_harabasz_score=result['ch_score'], 
                            silhouette_score=result['sil_score'],
                            number_of_clusters=number_of_clusters,
                            total_records=number_of_records,
                            word2vec_vector_size=word2vec_vector_size,
                            word2vec_window_size=word2vec_window_size,
                            word2vec_word_min_count_percentage=word2vec_word_min_count_percentage,
                            from_date = start_date,
                            applied=True,
                            inertia = result['inertia'])

    first_row.training_thread_running = False
    first_row.save()
    return





def test_n_clusters(data):




def test_n_clusters(start_clusters, 
                    end_clusters, 
                    step, 
                    start_date,
                    word2vec_vector_size,
                    word2vec_window_size,
                    word2vec_word_min_count_percentage):
 
    first_row = globals.objects.first()
    first_row.testing_thread_running = True
    first_row.save()
    job_posts = Job_Post.objects.filter(added_date__gte=start_date)
    text_data = list(job_posts.values_list('clusterable_text', flat=True))
        
    ret = test_number_of_clusters_gensim_kmeans(text_data=text_data,
                                          word2vec_vector_size = word2vec_vector_size,
                                          word2vec_window_size = word2vec_window_size,  
                                          start_number=start_clusters, 
                                          end_number=end_clusters, 
                                          step=step,
                                          word2vec_word_min_count_percentage=word2vec_word_min_count_percentage)

    results=ret['results']
    for record in results:
        cluster_records.objects.create(
                            calinski_harabasz_score = record['ch_score'], 
                            silhouette_score = record['sil_score'],
                            number_of_clusters = record['n_clusters'],
                            total_records = len(text_data),
                            word2vec_vector_size = word2vec_vector_size,
                            word2vec_window_size = word2vec_window_size,
                            word2vec_word_min_count_percentage = record['word2vec_word_min_count_percentage'],
                            from_date = start_date,
                            applied = False,
                            inertia = record['inertia']
        )
    
    first_row.number_of_rows = len(results)
    first_row.testing_thread_running=False
    first_row.save()
    return 
  
"""