# Databricks notebook source
cluster_info_cols = [
    'spark.databricks.clusterUsageTags.clusterNodeType', 
    'spark.databricks.clusterUsageTags.clusterMaxWorkers', 
    'spark.databricks.clusterUsageTags.clusterMinWorkers'
]
cluster_config = {el[0].replace('spark.databricks.clusterUsageTags.', ''): el[1] for el in spark.sparkContext.getConf().getAll() if el[0] in cluster_info_cols}

# COMMAND ----------

if cluster_config['clusterNodeType']=='Standard_DS5_v2':
    cluster_config['GB'] = 56
    cluster_config['core'] = 16
elif cluster_config['clusterNodeType']=='Standard_DS3_v2':
    cluster_config['GB'] = 14
    cluster_config['core'] = 4
elif cluster_config['clusterNodeType']=="dev-tier-node":
    cluster_config['GB'] = 15.3
    cluster_config['core'] = 2
    
del cluster_config['clusterNodeType']

# COMMAND ----------

# MAGIC %md # Boston data

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

# COMMAND ----------

boston = load_boston()

boston_pd = pd.DataFrame(
    data= np.c_[boston['data'], boston['target']],
    columns= np.append(boston['feature_names'], 'target')
)
boston_pd.head(5), boston_pd.shape[0]

# COMMAND ----------

# MAGIC %md # RF - sklearn

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# COMMAND ----------

train, test = train_test_split(boston_pd, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md Sztuczne powiększenie danych - tylko dla celów testowych

# COMMAND ----------

n = train.shape[0] * 1 #1000

# COMMAND ----------

train = train.sample(n, replace=True)

# COMMAND ----------

y_train = train['target']
X_train = train.drop(['target'], axis=1)

y_test = test['target']
X_test = test.drop(['target'], axis=1)

# rf = RandomForestRegressor()
# model = rf.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# print("MAE: " + str(mae))

# COMMAND ----------

# MAGIC %md #RF with tracking

# COMMAND ----------

seed = 42  # for repeatability purposes
default_params = {'n_estimators': 200, 'max_depth': 10}

# COMMAND ----------

import time
import mlflow
import mlflow.sklearn

# COMMAND ----------

mlflow.start_run(run_name="sklearn")

rf = RandomForestRegressor(n_estimators=default_params['n_estimators'], max_depth=default_params['max_depth'], random_state=seed)

start = time.time()
model = rf.fit(X_train, y_train)
end = time.time()

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mlflow.log_metrics({
    "time": end-start,
    "mae": mae,
})
mlflow.log_params(cluster_config)
mlflow.log_param("n", X_train.shape[0])
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

# COMMAND ----------

# features importances
pd.DataFrame({'var': X_train.columns, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)

# COMMAND ----------

def save_model(save, run_name, cluster_config, time, mae, n, cv, parallelism, model, lib):
    if save:
        mlflow.start_run(run_name=run_name)

        mlflow.log_metrics({
            "time": time,
            "mae": mae,
        })
        mlflow.log_params(cluster_config)
        mlflow.log_params({"n": n, 'cv': cv, 'parallelism': parallelism})
        lib.log_model(model, "model")

        mlflow.end_run()

# COMMAND ----------

save = True

# COMMAND ----------

# MAGIC %md #MLlib
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/ml-classification-regression.html

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor as RandomForestRegressorSpark
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

import mlflow.spark

# COMMAND ----------

train_spark = spark.createDataFrame(train).withColumnRenamed('target', 'label')
test_spark = spark.createDataFrame(test)

# COMMAND ----------

assembler = VectorAssembler(inputCols= list(X_train.columns), outputCol="features")
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html#pyspark.ml.regression.RandomForestRegressor
rf = RandomForestRegressorSpark(numTrees=default_params['n_estimators'], maxDepth=default_params['max_depth'], seed=seed)


pipeline = Pipeline(stages = [assembler, rf])

start = time.time()
model = pipeline.fit(train_spark)
end = time.time()

test_pred_spark = model.transform(test_spark)

evaluator = RegressionEvaluator(labelCol='target', predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(test_pred_spark)

# COMMAND ----------

save_model(save, "mllib", cluster_config, end-start, mae, train_spark.count(), None, None, model, mlflow.spark)

# COMMAND ----------

# MAGIC %md #Grid search

# COMMAND ----------

cv = 10
parameters = {'n_estimators': [10, 100, 200], 'max_depth': [5, 10, 15], 'random_state':[42]}

# COMMAND ----------

# MAGIC %md ## Grid search with cross validation - sklearn

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# COMMAND ----------

rf = RandomForestRegressor()
gs = GridSearchCV(rf, parameters, cv=cv, scoring='neg_mean_absolute_error')
start = time.time()
gs.fit(X_train, y_train)
end = time.time()
end - start

save_model(save, "sklearn-grid", cluster_config, end-start, abs(gs.best_score_), X_train.shape[0], cv, None, gs.best_estimator_, mlflow.sklearn)

# COMMAND ----------

#pd.DataFrame(gs.cv_results_)

# COMMAND ----------

# MAGIC %md ## Grid search with parallelism

# COMMAND ----------

parallelism = 4

# COMMAND ----------

from sklearn.utils import parallel_backend
!pip install joblibspark
from joblibspark import register_spark

# COMMAND ----------

register_spark() # register spark backend

start = time.time()
with parallel_backend("spark", n_jobs=parallelism):
    gs.fit(X_train, y_train)
end = time.time()

# COMMAND ----------

gs.best_params_

# COMMAND ----------

save_model(save, "sklearn-grid-parallel", cluster_config, end-start, abs(gs.best_score_), X_train.shape[0], cv, parallelism, gs.best_estimator_, mlflow.sklearn)

# COMMAND ----------

# MAGIC %md ##Grid search with cross validation - MLlib

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder

# COMMAND ----------

rf = RandomForestRegressorSpark()
pipeline = Pipeline(stages = [assembler, rf])
evaluator = RegressionEvaluator(labelCol='label', predictionCol="prediction", metricName="mae")

paramGrid = ParamGridBuilder()\
    .addGrid(rf.maxDepth, parameters['max_depth'])\
    .addGrid(rf.numTrees, parameters['n_estimators'])\
    .addGrid(rf.seed, parameters['random_state'])\
    .build()

cvSpark = CrossValidator(
    estimator=pipeline, 
    evaluator=evaluator,
    estimatorParamMaps=paramGrid,
    numFolds=cv,
    parallelism=parallelism,
    seed=seed
)

#pipeline = Pipeline(stages = [assembler, rf, cvSpark])

start = time.time()
cvModel = cvSpark.fit(train_spark)
end = time.time()

# COMMAND ----------

cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ].values()

# COMMAND ----------

save_model(save, "mllib-grid-parallel", cluster_config, end-start, min(cvModel.avgMetrics), train_spark.count(), cv, parallelism, cvModel.bestModel, mlflow.spark)

# COMMAND ----------

# MAGIC %md # H20
# MAGIC 
# MAGIC https://pypi.org/project/h2o-pysparkling-3.1/
# MAGIC 
# MAGIC http://docs.h2o.ai/sparkling-water/3.1/latest-stable/doc/pysparkling.html
# MAGIC 
# MAGIC https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html

# COMMAND ----------

from pysparkling import *
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

# COMMAND ----------

import mlflow.h2o  # https://www.mlflow.org/docs/latest/python_api/mlflow.h2o.html

# COMMAND ----------

hc = H2OContext.getOrCreate();

# COMMAND ----------

train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

# COMMAND ----------

rf_h2o = H2ORandomForestEstimator(
    ntrees=default_params['n_estimators'],
    max_depth=default_params['max_depth'],
    seed=seed
)

start = time.time()
rf_h2o.train(list(X_train.columns), 'target', training_frame=train_h2o, validation_frame=test_h2o)
end = time.time()

# COMMAND ----------

#rf_h2o.varimp()

# COMMAND ----------

save_model(save, "h2o", cluster_config, end-start, rf_h2o.mae(valid=True), train_h2o.nrow, None, None, rf_h2o, spark.h2o)

# COMMAND ----------

# MAGIC %md # Grid search with H2O

# COMMAND ----------

from h2o.grid.grid_search import H2OGridSearch

# COMMAND ----------

parameters_h2o = parameters.copy()
parameters_h2o['ntrees'] = parameters_h2o['n_estimators']
parameters_h2o['seed'] = parameters_h2o['random_state']
del parameters_h2o['n_estimators']
del parameters_h2o['random_state']

parameters_h2o, parameters

# COMMAND ----------

gs_h2o = H2OGridSearch(
    model=H2ORandomForestEstimator,
    hyper_params=parameters_h2o,
    parallelism=parallelism,
   # search_criteria={'strategy': 'Cartesian', 'stopping_metric': 'mae'}
)

start = time.time()
gs_h2o.train(list(X_train.columns), 'target', training_frame=train_h2o, seed=seed)
end = time.time()

best_gs_h2o = gs_h2o.get_grid(sort_by='mae', decreasing=False).models[0]

# COMMAND ----------

best_gs_h2o.params['ntrees']

# COMMAND ----------

save_model(save, "h2o-grid", cluster_config, end-start, best_gs_h2o.mae(test_h2o), train_h2o.nrow, cv, parallelism, best_gs_h2o , mlflow.h2o)

# COMMAND ----------

# MAGIC %md # Koalas vs Pandas
# MAGIC 
# MAGIC https://koalas.readthedocs.io/en/latest/

# COMMAND ----------

import databricks.koalas as ks

# COMMAND ----------

train_cat = train.copy()
train_cat['AGE_CAT'] = pd.cut(train_cat['AGE'], bins=3, labels=['one', 'two', 'three']).astype(str)

# COMMAND ----------

train_ks = ks.from_pandas(train_cat)
test_ks = ks.from_pandas(test)

# COMMAND ----------

train_ks.head()

# COMMAND ----------

ks.sql("SELECT * FROM {train_ks} WHERE AGE > 80").shape

# COMMAND ----------

train_ks.groupby(['AGE_CAT']).count()

# COMMAND ----------

train_cat.groupby(['AGE_CAT']).count()

# COMMAND ----------

# MAGIC %md # Użyteczne transformacje

# COMMAND ----------

train_cat_spark = spark.createDataFrame(train_cat)

# COMMAND ----------

# MAGIC %md ## Window

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql import functions as f

# COMMAND ----------

window = Window.partitionBy('AGE_CAT')

train_cat_spark\
    .withColumn('avg_age', f.rank().over(window.orderBy(f.asc('AGE'))))\
    .display()

# COMMAND ----------

# MAGIC %md ## ApplyInPandas

# COMMAND ----------

from sklearn.model_selection import cross_val_score

# COMMAND ----------

from pyspark.sql.types import *

# COMMAND ----------

def train_rf(data):
    rf = RandomForestRegressor()
    
    y_train = data['target']
    X_train = data.drop(['target', 'AGE_CAT'], axis=1)
    
    mae = cross_val_score(rf, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
    print(mae)
    return pd.DataFrame({'fold': range(cv), 'mae': mae, 'age': data['AGE_CAT'].max()})

# COMMAND ----------

train_rf(train_cat)

# COMMAND ----------

# https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.types.StructField.html
schema = StructType([
    StructField('fold', IntegerType(), True),
    StructField('mae', DoubleType(), True),
    StructField('age', StringType(), True),
])  

# COMMAND ----------

res = train_cat_spark\
    .groupBy('AGE_CAT')\
    .applyInPandas(train_rf, schema)

res.display()

# COMMAND ----------

res.display()

# COMMAND ----------

res.cache()

# COMMAND ----------

res.display()

# COMMAND ----------

res.display()

# COMMAND ----------

res\
    .groupBy('age')\
    .pivot('fold')\
    .agg(f.avg('mae').alias('mae'))\
    .display()

# COMMAND ----------

# MAGIC %md ## Reduce

# COMMAND ----------

from pyspark.sql import DataFrame # https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.html?highlight=datafram

# COMMAND ----------

from functools import reduce

# COMMAND ----------

train, test = train_test_split(boston_pd, test_size=0.2, random_state=42)

# COMMAND ----------

train_spark = spark.createDataFrame(train)
test_spark = spark.createDataFrame(test)

# COMMAND ----------

train_array = [train_spark, test_spark]
all_data = reduce(DataFrame.unionAll, train_array)
all_data.count()

# COMMAND ----------

all_data.toPandas().head()

# COMMAND ----------

# MAGIC %md ## UDF

# COMMAND ----------

import math

# COMMAND ----------

@udf(returnType=DoubleType()) 
def udf_sqrt(x):
    return math.sqrt(x)

# COMMAND ----------

all_data.withColumn('x2', udf_sqrt('TAX')).limit(2).display()

# COMMAND ----------

all_data.withColumn('x2', f.sqrt('TAX')).limit(2).display()

# COMMAND ----------

all_data.write.mode('overwrite').parquet('/FileStore/test.parquet')

# https://community.cloud.databricks.com/?o=2097898880805746#tables/new/dbfs
# https://mungingdata.com/apache-spark/partitionby/#:~:text=partitionBy()%20is%20a%20DataFrameWriter,important%20independent%20of%20disk%20partitioning.

# COMMAND ----------

# MAGIC %md ## Expanding

# COMMAND ----------

train_cat_spark\
    .withColumn('age_q', f.expr('percentile(AGE, array(0.0, 0.5, 1.0))').over(Window.partitionBy('AGE_CAT')))\
    .withColumn("q", f.array(f.lit(0), f.lit(0.5), f.lit(1.0)))\
    .withColumn("zipped", f.arrays_zip(*(["q"] + ['age_q'])))\
    .drop(*(["q"]))\
    .withColumn("zipped", f.explode("zipped"))\
    .select("AGE_CAT", "zipped.*").drop("zipped")\
    .dropDuplicates()\
    .orderBy('AGE_CAT', 'q')\
    .display()

# COMMAND ----------

# MAGIC %md #Other languages

# COMMAND ----------

# MAGIC %md ## Scala

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC var data_spark = spark.read.parquet("/FileStore/test.parquet")
# MAGIC 
# MAGIC display(data_spark)

# COMMAND ----------

# MAGIC %md ## R
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/sparkr.html#:~:text=SparkR%20is%20an%20R%20package,dplyr)%20but%20on%20large%20datasets.

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)

# COMMAND ----------

# MAGIC %r
# MAGIC data_r <- read.parquet("/FileStore/test.parquet")
# MAGIC head(data_r)

# COMMAND ----------

# MAGIC %r
# MAGIC head(select(data_r, "CRIM"))

# COMMAND ----------

# MAGIC %md ## SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE test

# COMMAND ----------

all_data.write.saveAsTable("test.boston")

# COMMAND ----------

sql('select * from test.boston').display()

# COMMAND ----------

# %sql
# drop DATABASE asdf

# COMMAND ----------

# MAGIC %md # Joins
# MAGIC 
# MAGIC ## Cross join

# COMMAND ----------

parameters = {'n_estimators': [10, 100, 200], 'max_depth': [5, 10, 15], 'random_state':[42]}

# COMMAND ----------

from pyspark.sql.types import FloatType
import pandas as pd

params_spark = spark.createDataFrame(pd.DataFrame({'n_estimators': [10, 100, 200]}))
params_spark.display()

# COMMAND ----------

sql('select * from test.boston').crossJoin(params_spark).count()

# COMMAND ----------

# MAGIC %md ## Join

# COMMAND ----------

data = sql('select * from test.boston').withColumn('id', f.monotonically_increasing_id())

data.select('id', 'CRIM').join(data.select('id', 'PTRATIO').limit(2), on='id', how='left').display()

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._ 
# MAGIC 
# MAGIC var data = sql("select * from test.boston").withColumn("id", monotonicallyIncreasingId)
# MAGIC 
# MAGIC var subset1 = data.select("id", "CRIM")
# MAGIC var subset2 = data.select("id", "PTRATIO").limit(2)
# MAGIC 
# MAGIC display(subset1.join(subset2, subset1("id") === subset2("id"), "inner"))

# COMMAND ----------

# MAGIC %md #Hyperopt
# MAGIC 
# MAGIC https://docs.databricks.com/_static/notebooks/hyperopt-spark-mlflow.html

# COMMAND ----------

from hyperopt import fmin, tpe, rand, hp, SparkTrials, Trials, STATUS_OK

# COMMAND ----------

# function to minimize
def objective(C):
    rf = RandomForestRegressor(C)
    
    mae = cross_val_score(rf, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
    
    return {'loss': -mae, 'status': STATUS_OK}

# search space
values = [5, 10, 50, 100, 500]
search_space = hp.choice('C', values)
# rand
algo=rand.suggest

# COMMAND ----------

# argmin = fmin(
#   fn=objective,
#   space=search_space,
#   algo=algo,
#   max_evals=3
# )
# values[argmin['C']]

# COMMAND ----------

spark_trials = SparkTrials(parallelism=4)
 
with mlflow.start_run():
    argmin = fmin(
        fn=objective,
        space=search_space,
        algo=algo,
        max_evals=3,
        trials=spark_trials
    )
# Print the best value found for C
print("Best value found: ", values[argmin['C']])

# COMMAND ----------

help(SparkTrials)

# COMMAND ----------


