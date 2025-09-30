import kagglehub

# Download datasets
iris_path = kagglehub.dataset_download("uciml/iris")
youtube_path = kagglehub.dataset_download("asaniczka/2024-youtube-channels-1-million") # Вариант 1
bike_use_path = kagglehub.dataset_download("sebastianquirarte/over-9-years-of-real-public-bike-use-data-mibici") # Вариант 3


print("Path to iris dataset files:", iris_path)
print("Path to youtube dataset files:", youtube_path)
print("Path to bike use dataset files:", bike_use_path)
