import kagglehub

# Download datasets
iris_path = kagglehub.dataset_download("uciml/iris")
youtube_path = kagglehub.dataset_download("asaniczka/2024-youtube-channels-1-million")

print("Path to iris dataset files:", iris_path)
print("Path to youtube dataset files:", youtube_path)
