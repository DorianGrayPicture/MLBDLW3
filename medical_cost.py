from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.sql.functions import col
import numpy as np

# Создаем Spark-сессию
spark = SparkSession.builder \
    .appName("InsuranceAnalysis") \
    .getOrCreate()

# Загружаем данные
df = spark.read.csv("/home/denis/.cache/kagglehub/datasets/mirichoi0218/insurance/versions/1/insurance.csv", header=True, inferSchema=True)

print("Первые 5 строк данных:")
df.show(5)

print("Схема данных:")
df.printSchema()

# 1. РЕГРЕССИЯ (предсказание charges)
print("="*50)
print("РЕГРЕССИОННЫЙ АНАЛИЗ")
print("="*50)

# Предобработка для регрессии
sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_index")
smoker_indexer = StringIndexer(inputCol="smoker", outputCol="smoker_index")
region_indexer = StringIndexer(inputCol="region", outputCol="region_index")
encoder = OneHotEncoder(inputCols=["region_index"], outputCols=["region_encoded"])

# Для регрессии используем все признаки, включая smoker
feature_cols_reg = ["age", "sex_index", "bmi", "children", "smoker_index", "region_encoded"]
assembler_reg = VectorAssembler(inputCols=feature_cols_reg, outputCol="features")

# Пайплайн для регрессии
pipeline_reg = Pipeline(stages=[
    sex_indexer,
    smoker_indexer,
    region_indexer,
    encoder,
    assembler_reg
])

# Обучаем пайплайн и преобразуем данные
processed_reg = pipeline_reg.fit(df).transform(df)

# Разделяем данные на train/test
train_data, test_data = processed_reg.randomSplit([0.7, 0.3], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="charges")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.evaluate(test_data)

print(f"RMSE: {lr_predictions.rootMeanSquaredError:.2f}")
print(f"R2: {lr_predictions.r2:.4f}")
print(f"MAE: {lr_predictions.meanAbsoluteError:.2f}")

# 2. КЛАССИФИКАЦИЯ (предсказание smoker)
print("\n" + "="*50)
print("КЛАССИФИКАЦИЯ")
print("="*50)

# Для классификации НЕ используем smoker в качестве признака!
feature_cols_class = ["age", "sex_index", "bmi", "children", "region_encoded"]
assembler_class = VectorAssembler(inputCols=feature_cols_class, outputCol="features_class")

# Пайплайн для классификации
pipeline_class = Pipeline(stages=[
    sex_indexer,
    region_indexer,
    encoder,
    assembler_class
])

# Обучаем пайплайн и преобразуем данные
processed_class = pipeline_class.fit(df).transform(df)

# Добавляем целевую переменную (индексированный smoker)
smoker_indexer_class = StringIndexer(inputCol="smoker", outputCol="label")
processed_class = smoker_indexer_class.fit(processed_class).transform(processed_class)

print("Признаки для классификации:")
processed_class.select("age", "sex", "bmi", "children", "region", "smoker", "label", "features_class").show(10)

# Разделяем данные на train/test
class_train, class_test = processed_class.randomSplit([0.7, 0.3], seed=42)

logreg = LogisticRegression(featuresCol="features_class", labelCol="label")
logreg_model = logreg.fit(class_train)
class_predictions = logreg_model.transform(class_test)

print("Примеры предсказаний классификации:")
class_predictions.select("age", "sex", "bmi", "smoker", "label", "prediction", "probability").show(10)

# Метрики для классификации
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(class_predictions)

print(f"Area under ROC: {auc:.4f}")

# Дополнительные метрики
tp = class_predictions.filter("prediction = 1.0 AND label = 1.0").count()
tn = class_predictions.filter("prediction = 0.0 AND label = 0.0").count()
fp = class_predictions.filter("prediction = 1.0 AND label = 0.0").count()
fn = class_predictions.filter("prediction = 0.0 AND label = 1.0").count()

total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("Матрица ошибок:")
print(f"True Positive: {tp}")
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"Всего примеров: {total}")

# Проверим распределение целевой переменной
print("\nРаспределение курильщиков в данных:")
class_test.groupBy("smoker", "label").count().show()

# 3. КЛАСТЕРИЗАЦИЯ
print("\n" + "="*50)
print("КЛАСТЕРИЗАЦИЯ")
print("="*50)

# Используем только числовые признаки для кластеризации
cluster_feature_cols = ["age", "bmi", "children"]
cluster_assembler = VectorAssembler(inputCols=cluster_feature_cols, outputCol="cluster_features")
cluster_df = cluster_assembler.transform(df)

kmeans = KMeans(featuresCol="cluster_features", k=3, seed=42)
kmeans_model = kmeans.fit(cluster_df)
cluster_predictions = kmeans_model.transform(cluster_df)

# Метрики кластеризации
evaluator = ClusteringEvaluator(featuresCol='cluster_features')
silhouette = evaluator.evaluate(cluster_predictions)

# Вычисляем WSSSE вручную
def calculate_wssse(model, data):
    centers = model.clusterCenters()
    wssse = data.rdd.map(lambda row: 
        np.sum((row['cluster_features'].toArray() - centers[int(row['prediction'])]) ** 2)
    ).sum()
    return wssse

wssse = calculate_wssse(kmeans_model, cluster_predictions)

print(f"Silhouette score: {silhouette:.4f}")
print(f"Within Set Sum of Squared Errors: {wssse:.2f}")

# Показываем центры кластеров
centers = kmeans_model.clusterCenters()
print("Cluster Centers (age, bmi, children):")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# Статистика по кластерам
print("\nРаспределение по кластерам:")
cluster_predictions.groupBy("prediction").count().orderBy("prediction").show()

# Останавливаем Spark-сессию
spark.stop()
