import kagglehub

# Download datasets
iris_path = kagglehub.dataset_download("uciml/iris")
youtube_path = kagglehub.dataset_download("asaniczka/2024-youtube-channels-1-million") # Вариант 1
bike_use_path = kagglehub.dataset_download("sebastianquirarte/over-9-years-of-real-public-bike-use-data-mibici") # Вариант 3
cars_path = kagglehub.dataset_download("jitikpatel/used-cars-price-dataset") # Вариант 6
avocado_path = kagglehub.dataset_download("neuromusic/avocado-prices") # Вариант 14
medical_cost_path = kagglehub.dataset_download("mirichoi0218/insurance") # Вариант 9

print("Path to iris dataset files:", iris_path)
print("Path to youtube dataset files:", youtube_path)
print("Path to bike use dataset files:", bike_use_path)
print("Path to cars dataset files:", cars_path)
print("Path to avocado dataset files:", avocado_path)
print("Path to medical cost dataset files:", medical_cost_path)
