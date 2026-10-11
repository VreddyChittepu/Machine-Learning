"""Generate standalone, explained Databricks notebooks; no external dependencies."""
from pathlib import Path
import ast
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'algorithms'
OUT.mkdir(parents=True, exist_ok=True)
SPECS = [
('02_linear_regression','Linear Regression','regression','LinearRegression','pyspark.ml.regression',[{'regParam':0.0,'fitIntercept':True},{'regParam':0.0,'fitIntercept':False}],{'maxIter':100},
 'Learns a weighted sum of inputs to predict a number. Prediction = intercept + sum(weight × measurement). Least squares penalizes large errors by squaring them. A wine laboratory can use it as a simple quality-score baseline.',
 'fitIntercept allows a constant offset; regParam=0 disables penalties; maxIter caps iterative solver work. Compare intercept on/off on validation data.',
 'It extends Lesson 1 from one feature to eleven. It is easy to interpret but cannot directly learn complicated feature interactions.'),
('03_ridge_regression','Ridge Regression','regression','LinearRegression','pyspark.ml.regression',[{'regParam':0.01},{'regParam':0.1}],{'elasticNetParam':0.0,'maxIter':100},
 'Adds an L2 penalty to linear regression: large squared weights cost more. It can stabilize predictions when chemical measurements overlap in information. A weight of 3 contributes 9 to the unscaled squared-weight penalty.',
 'regParam sets penalty strength; elasticNetParam=0 selects L2; maxIter is an iteration cap. The two settings are a small demonstration grid, not an exhaustive search.',
 'Compared with unregularized linear regression, Ridge shrinks coefficients; it usually does not make them exactly zero.'),
('04_lasso_regression','Lasso Regression','regression','LinearRegression','pyspark.ml.regression',[{'regParam':0.01},{'regParam':0.1}],{'elasticNetParam':1.0,'maxIter':100},
 'Adds an L1 penalty, based on absolute weights. A weight of -3 contributes 3 to the unscaled absolute-weight penalty. Some coefficients can become zero, creating a simpler predictive formula.',
 'regParam sets penalty strength; elasticNetParam=1 selects L1; maxIter caps optimization iterations.',
 'Unlike Ridge, Lasso can zero coefficients. Correlated inputs can make the selected feature set unstable; zero weight is not proof that a chemical has no physical effect.'),
('05_decision_tree_regression','Decision Tree Regression','regression','DecisionTreeRegressor','pyspark.ml.regression',[{'maxDepth':3},{'maxDepth':5}],{'seed':42,'minInstancesPerNode':5},
 'Learns a sequence of questions such as alcohol <= 10.5, then predicts the average training label within the reached leaf. This models nonlinear rules useful for a quality-control prototype.',
 'maxDepth limits consecutive splits; minInstancesPerNode=5 prevents tiny leaves; seed fixes stochastic choices where used.',
 'Unlike a linear model, a tree learns thresholds and interactions. A deep tree can memorize training data and become unstable.'),
('06_random_forest_regression','Random Forest Regression','regression','RandomForestRegressor','pyspark.ml.regression',[{'maxDepth':4},{'maxDepth':6}],{'seed':42,'numTrees':30,'minInstancesPerNode':5},
 'Averages predictions from many randomized trees. If three trees predict 5, 6 and 7, their average is 6. Randomized sampling and feature choices help reduce individual-tree instability.',
 'numTrees=30 limits this learning exercise to a modest forest; maxDepth controls each tree; minInstancesPerNode limits small leaves; seed aids reproducibility.',
 'Usually more stable than one tree, with greater memory and training costs. It still needs honest holdout evaluation.'),
('07_gradient_boosted_tree_regression','Gradient Boosted Tree Regression','regression','GBTRegressor','pyspark.ml.regression',[{'maxDepth':2},{'maxDepth':4}],{'seed':42,'maxIter':20,'stepSize':0.1},
 'Builds trees sequentially to correct the current model errors. For intuition, prediction 5 plus a 0.1 learning-rate correction of 2 becomes 5.2. It is useful for nonlinear tabular prediction.',
 'maxIter=20 limits boosting rounds; stepSize=0.1 shrinks each update; maxDepth controls tree complexity; seed aids reproducibility.',
 'Random forests average randomized trees; boosting adds sequential corrections. More rounds and depth can overfit.'),
('08_logistic_regression','Logistic Regression','classification','LogisticRegression','pyspark.ml.classification',[{'regParam':0.01},{'regParam':0.1}],{'elasticNetParam':0.0,'maxIter':100,'threshold':0.5},
 'Predicts a probability for a binary outcome using a sigmoid: p=1/(1+exp(-z)). A score z=0 gives p=0.5. Here the teaching outcome is high quality (score >=7), not a real commercial certification.',
 'regParam is L2 strength because elasticNetParam=0; maxIter caps optimization; threshold=0.5 maps probability to label. It is fixed here, not tuned on test data.',
 'Despite its name, this is classification. Linear regression predicts an unrestricted number; logistic regression models a binary probability.'),
('09_decision_tree_classification','Decision Tree Classification','classification','DecisionTreeClassifier','pyspark.ml.classification',[{'maxDepth':3},{'maxDepth':5}],{'seed':42,'minInstancesPerNode':5},
 'Asks feature-threshold questions to separate label 0 and label 1. Leaf class distributions produce probability estimates and a predicted class. A small tree offers understandable screening rules.',
 'maxDepth controls rule complexity; minInstancesPerNode prevents very small leaves; seed controls randomized choices where applicable.',
 'Unlike logistic regression, it learns nonlinear thresholds without specifying interactions manually. Its leaf probabilities can be poorly calibrated.'),
('10_random_forest_classification','Random Forest Classification','classification','RandomForestClassifier','pyspark.ml.classification',[{'maxDepth':4},{'maxDepth':6}],{'seed':42,'numTrees':30,'minInstancesPerNode':5},
 'Combines randomized classification trees. Their class estimates form an ensemble prediction, improving stability over an individual tree. It is a useful tabular screening baseline.',
 'numTrees sets ensemble size; maxDepth sets tree complexity; minInstancesPerNode constrains leaves; seed aids repeatability.',
 'More stable than a single tree, generally less interpretable than logistic regression. Evaluate minority-class recall rather than relying on accuracy.'),
('11_gradient_boosted_tree_classification','Gradient Boosted Tree Classification','classification','GBTClassifier','pyspark.ml.classification',[{'maxDepth':2},{'maxDepth':4}],{'seed':42,'maxIter':20,'stepSize':0.1},
 'Sequential trees improve a binary classification loss. Each new tree corrects weaknesses in the current ensemble. This targets label 1 for score >=7 and label 0 otherwise.',
 'maxIter controls boosting rounds; stepSize shrinks updates; maxDepth limits tree complexity; seed aids repeatability. Spark GBTClassifier supports binary classification.',
 'Compared with a random forest, boosting is sequential and can be more sensitive to settings. It is not XGBoost; they are different implementations.'),
('12_kmeans_clustering','K-Means Clustering','clustering','KMeans','pyspark.ml.clustering',[{'k':2},{'k':3}],{'seed':42,'maxIter':30},
 'Groups similar chemical profiles without quality labels. It repeatedly assigns each point to its nearest centre and recomputes the centres. For points 2 and 4 in one dimension the centre is 3.',
 'k chooses the number of groups; maxIter caps assignment/update rounds; seed controls initialization. Euclidean distances depend strongly on scaling.',
 'Unlike supervised models, K-Means does not learn quality. Cluster numbers are arbitrary identifiers and cannot be interpreted as good/bad wine.'),
('13_pca_dimension_reduction','Principal Component Analysis','pca','PCA','pyspark.ml.feature',[{'k':2},{'k':5}],{},
 'Compresses correlated measurements into orthogonal directions of variation. If two standardized measurements move together, a component can summarize their shared direction. This supports visualization and downstream feature compression.',
 'k is the retained component count; inputCol identifies standardized features; outputCol names the compressed vector. PCA has no label or predictionCol.',
 'PCA creates continuous components; K-Means creates groups. PCA is not a classifier, clusterer or proof that the retained directions predict quality.'),
]

LOAD = '''
from functools import reduce
from uuid import uuid4
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, ClusteringEvaluator
import mlflow
import math

SEED = 42
DATA_PATH = "/databricks-datasets/wine-quality/winequality-red.csv"
# If blocked, set this to the SAME CSV copied into your accessible UC volume.
DATA_PATH_OVERRIDE = ""
source_path = DATA_PATH_OVERRIDE or DATA_PATH
raw = spark.read.option("header", True).option("sep", ";").csv(source_path)
raw.limit(1).collect()  # Force an early access check; only one row reaches Python.
raw = raw.toDF(*[name.strip().replace(" ", "_") for name in raw.columns])
feature_columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH",
    "sulphates", "alcohol"]
required = feature_columns + ["quality"]
assert set(required).issubset(raw.columns), "Check the CSV path and semicolon separator"
numeric = raw.select(*[F.expr(f"try_cast(`{c}` AS DOUBLE)").alias(c) for c in required])
print("Source:", source_path)
print("Raw rows:", numeric.count())
numeric.printSchema()
'''
CLEAN = '''
def finite_or_null(name):
    value = F.col(name)
    return F.when(value.isNull() | F.isnan(value) | (F.abs(value) == float("inf")),
                  F.lit(None).cast("double")).otherwise(value)

normalized = numeric.select(*[finite_or_null(c).alias(c) for c in required])
label_valid = F.col("quality").between(0.0, 10.0) & (F.col("quality") == F.floor("quality"))
quarantine = normalized.filter(F.col("quality").isNull() | ~label_valid)
valid = normalized.filter(label_valid)
clean = valid.dropDuplicates(required)
print("Rejected labels:", quarantine.count())
print("Duplicate rows removed:", valid.count() - clean.count())
print("Clean rows:", clean.count())
clean = clean.withColumn("record_id", F.sha2(F.to_json(F.struct(*[F.col(c) for c in required])), 256))
# Identical feature profiles always share a split, even if their labels differ.
clean = clean.withColumn("split_bucket", F.pmod(F.xxhash64(*feature_columns), F.lit(100)))
'''
SPLIT = '''
train = data.filter(F.col("split_bucket") < 70)
validation = data.filter(F.col("split_bucket").between(70, 84))
test = data.filter(F.col("split_bucket") >= 85)
for name, frame in [("train", train), ("validation", validation), ("test", test)]:
    assert frame.count() > 0, f"Empty {name} split"
    print(name, frame.count())
missing = train.select(*[F.count(c).alias(c) for c in feature_columns]).first().asDict()
assert all(count > 0 for count in missing.values()), "A training feature is entirely null"
train.select(*feature_columns).describe().show(truncate=False)
'''
REG_METRICS = '''
def evaluate(frame, fitted=None):
    return {metric: RegressionEvaluator(labelCol="label", predictionCol="prediction",
                    metricName=metric).evaluate(frame) for metric in ["mae", "mse", "rmse", "r2"]}
selection_metric = "rmse"
maximize = False
'''
CLS_METRICS = '''
def evaluate(frame, fitted=None):
    a, p = F.col("label"), F.col("prediction")
    row = frame.agg(*[
        F.sum(F.when(condition, 1).otherwise(0)).alias(name)
        for name, condition in [
            ("tp", (a == 1) & (p == 1)), ("tn", (a == 0) & (p == 0)),
            ("fp", (a == 0) & (p == 1)), ("fn", (a == 1) & (p == 0))]
    ]).first().asDict()
    tp, tn, fp, fn = [row[c] for c in ["tp", "tn", "fp", "fn"]]
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {**row, "accuracy": (tp+tn)/(tp+tn+fp+fn), "precision": precision,
        "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
        "roc_auc": BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                                 metricName="areaUnderROC").evaluate(frame),
        "pr_auc": BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                                metricName="areaUnderPR").evaluate(frame)}
selection_metric = "pr_auc"
maximize = True
'''
CLUSTER_METRICS = '''
def evaluate(frame, fitted=None):
    groups = frame.select("prediction").distinct().count()
    if groups < 2 or groups >= frame.count():
        raise ValueError("Silhouette requires at least two nontrivial predicted groups")
    score = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
        metricName="silhouette", distanceMeasure="squaredEuclidean").evaluate(frame)
    return {"silhouette": score}
selection_metric = "silhouette"
maximize = True
'''
PCA_METRICS = '''
def evaluate(frame, fitted):
    # Orthogonal projection residual: ||x||² - ||projection(x)||².
    squared_input = F.aggregate(vector_to_array("features"), F.lit(0.0), lambda s, x: s + x*x)
    squared_projection = F.aggregate(vector_to_array("pca_features"), F.lit(0.0), lambda s, x: s + x*x)
    mse = frame.select(F.avg(F.greatest(squared_input-squared_projection, F.lit(0.0)) /
                             F.lit(len(feature_columns))).alias("mse")).first()["mse"]
    return {"reconstruction_mse": mse,
            "training_explained_variance": float(sum(fitted.stages[-1].explainedVariance))}
selection_metric = "reconstruction_mse"
maximize = False
'''
SAVE = '''
namespace = spark.sql("SELECT current_catalog() AS c, current_schema() AS s").first()
CATALOG, SCHEMA = namespace.c, namespace.s  # Choose a writable namespace if needed.
VOLUME = "ml_learning_models"
def quote_identifier(value):
    return "`" + value.replace("`", "``") + "`"
prefix = quote_identifier(CATALOG) + "." + quote_identifier(SCHEMA)
volume_identifier = prefix + "." + quote_identifier(VOLUME)
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_identifier}")
run_suffix = uuid4().hex[:12]
model_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{ALGORITHM}/{run_suffix}/pipeline"
source_table = prefix + "." + quote_identifier(ALGORITHM + "_source_" + run_suffix)
prediction_table = prefix + "." + quote_identifier(ALGORITHM + "_predictions_" + run_suffix)
data.write.format("delta").mode("errorifexists").saveAsTable(source_table)
best_model.write().save(model_path)
loaded_model = PipelineModel.load(model_path)
with mlflow.start_run(run_id=selected_run_id):
    mlflow.log_params({"model_path": model_path, "source_table": source_table})
    mlflow.log_dict({"model_format": "Spark PipelineModel", "model_path": model_path,
        "source_table": source_table, "features_in_order": feature_columns,
        "source_path": source_path, "algorithm": ALGORITHM}, "model_manifest.json")
print("Model path:", model_path)
print("Source table:", source_table)
'''

METRIC_TEXT = {
'regression': '''Error = actual − prediction. Actual [10,20,30], predicted [12,18,33] gives absolute errors [2,2,3]. MAE=7/3=2.333; MSE=17/3=5.667; RMSE=sqrt(17/3)=2.380; R²=1−17/200=0.915. Lower errors are better. R² can be negative and is not percentage accuracy. Quality is ordinal but treated as numerical for this exercise. MAE is measured in quality-score points; MSE uses squared points.''',
'classification': '''Positive class means quality >=7. TP=8, FP=2, FN=4, TN=86 gives accuracy=94/100=0.94, precision=8/10=0.80, recall=8/12=0.667, F1=16/22=0.727. Precision asks how many predicted positives were correct; recall asks how many actual positives were found. F1 balances both. We compute positive-class F1 explicitly, not Spark's weighted multiclass F1. Undefined precision/recall denominators return 0 by convention. ROC-AUC measures ranking across thresholds; for four positive/negative pairs, three correctly ordered pairs and no ties give AUC=0.75. PR-AUC emphasizes positive retrieval and is our validation selection metric because high-quality wines are less common. A high accuracy can hide zero positive recall. Threshold 0.5 is held fixed; tune thresholds on validation only in a later exercise.''',
'clustering': '''Silhouette compares within-cluster distance a with nearest-other-cluster distance b: (b−a)/max(a,b). If a=2 and b=5, the value is 0.6. Values near 1 indicate separation; near 0 indicate overlap; negative values suggest a closer alternative cluster. Spark's evaluator here uses squared Euclidean distance. We score held-out assignments using the training-fitted scaler and centres. Silhouette is not accuracy; business usefulness and cluster stability also matter.''',
'pca': '''Explained variance reports how much training-set variation each component represents. Ratios [0.40,0.25] retain 65%, not 65% prediction accuracy. Reconstruction MSE measures average squared distance lost when compressing standardized features. For x=[3,4] and projection=[3], discarded squared length is 25−9=16; divided by two input features gives MSE=8. More components generally lower reconstruction error, so we apply a small explicit complexity penalty in selection instead of blindly declaring the largest k best. The training-fitted centering/scaling is preserved in the pipeline.'''
}

for slug,title,kind,klass,module,configs,fixed,intro,param_text,comparison in SPECS:
    cells=[]
    def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(True)})
    def cell(section,why,details,expected,errors,src):
        md(f'## {section}\n\n**What and why:** {why}\n\n**Functions, important lines and parameters:** {details}\n\n**Expected output and interpretation:** {expected}\n\n**Common errors and debugging:** {errors}')
        src=textwrap.dedent(src).strip()+'\n'
        ast.parse(src)
        cells.append({'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':src.splitlines(True)})
    md(f'''# {title} — Databricks Wine Quality

Standalone notebook, one algorithm. Run cells in order. No pandas or RDDs. All metric values shown in explanations are illustrative unless described as mathematical examples. No training has been executed in the author's environment; your notebook prints actual metrics.

## 1. Business problem
{'Estimate the wine quality score from laboratory chemical measurements.' if kind=='regression' else 'Flag wines with quality score >=7 from chemical measurements; this is a teaching definition.' if kind=='classification' else 'Explore groups of similar chemical profiles, without quality labels.' if kind=='clustering' else 'Compress eleven correlated chemical measurements into fewer numerical components, without quality labels.'}

## 2. Algorithm introduction
{intro}

## 3. Real-world use case and comparison
{comparison}

## 4. Dataset selection and environment
Use `/databricks-datasets/wine-quality/winequality-red.csv`, documented by Databricks and based on the UCI Wine Quality dataset (Cortez et al., 2009). It contains eleven physicochemical features and a sensory quality score. Wine chemistry is not a customer-churn dataset; these notebooks teach reusable modelling patterns. The label is never part of the model input.

The standard red-wine file commonly contains 1,599 rows before deduplication; check the count printed here rather than assuming the workspace copy is identical. Semicolon is the separator. We use the same dataset and deterministic feature-profile split across algorithms, permitting meaningful within-task comparisons.

**Free Edition:** select a recent environment supporting these Spark ML operations. No `sparkContext`, RDD, `cache`, `persist`, or mounts are used. Do not pip-install PySpark. Add MLflow through notebook Environment dependencies if needed. Dataset access and model persistence depend on workspace configuration. If the legacy dataset path is blocked, inspect `/Volumes/samples/databricks/datasets/` for an available copy, or upload the same UCI red-wine CSV to an accessible Unity Catalog volume and set `DATA_PATH_OVERRIDE`. Do not substitute a synthetic file silently. An unavailable Spark ML API requires a supported environment; this notebook does not claim a fallback implementation for every model. Persistence requires a writable catalog/schema and CREATE VOLUME/TABLE plus file permissions.

Sources:
- https://docs.databricks.com/aws/en/getting-started/ml-get-started
- https://docs.databricks.com/aws/en/discover/databricks-datasets
- https://docs.databricks.com/aws/en/compute/serverless/limitations
- https://spark.apache.org/docs/latest/ml-classification-regression.html
- https://spark.apache.org/docs/latest/ml-clustering.html
- https://spark.apache.org/docs/latest/ml-features.html
''')
    cell('5. Data loading using PySpark','Read the requested real sample data with a reproducible schema contract.',
         '`spark.read` builds a CSV reader; header=True uses column names; sep=";" is essential. limit(1).collect() forces an early access check with only one record. toDF normalizes column names. try_cast produces NULL for malformed numeric strings even with ANSI mode. required declares the expected columns; assert catches the wrong dataset. No full dataset is collected. Imports provide Spark transformations, pipeline components, metrics and MLflow; uuid4 later gives unique output paths.',
         'Source path, row count and twelve double columns. Malformed numeric fields become missing values for later handling.',
         'PATH_NOT_FOUND or permission denial: verify the source in your workspace and set the override to the same CSV in a permitted volume. One giant column means the separator is wrong. Missing MLflow: add the dependency and restart/run from the top.',LOAD)
    md('''## 6. Exploratory data analysis
Training-only EDA runs after the split in section 9 so validation/test distributions do not influence feature choices. First we validate types and deterministic domain rules. We will inspect count, mean, standard deviation, minimum/maximum and training label balance. A histogram in Databricks display can reveal unusual ranges. Do not choose transformations by examining test outcomes.''')
    cell('7. Data cleaning','Handle invalid values, remove invalid labels and identical full records without learning from held-out data.',
         'finite_or_null converts NULL, NaN and infinities to NULL using when/otherwise; feature NULLs will be imputed from training only. between(0,10) and floor restrict quality to integer scores in its domain; missing/invalid labels go to quarantine. dropDuplicates(required) removes only exact duplicate feature+label rows, not all wines with the same score. sha2(to_json(struct(...)),256) identifies each complete record. xxhash64(feature_columns) and pmod(...,100) assign a stable bucket using features only. Identical feature profiles remain in one split, avoiding duplicate leakage.',
         'Rejected-label, duplicate and remaining counts. record_id is a lineage identifier, never a predictor. Hash splitting gives approximate proportions, not exact 70/15/15 counts.',
         'Do not fill missing labels with zero. Entirely missing training columns cannot be median-imputed; section 9 checks them. These generic rules do not replace chemistry-specific range review.',CLEAN)
    data_code = 'data = clean.withColumn("label", F.col("quality"))' if kind=='regression' else 'data = clean.withColumn("label", (F.col("quality") >= 7).cast("double"))' if kind=='classification' else 'data = clean.drop("quality")'
    cell('8. Feature engineering','Declare input columns and task output while excluding the answer and identifiers from features.',
         f'feature_columns contains only eleven chemical inputs. {"withColumn creates the numerical target label." if kind in ["regression","classification"] else "drop removes quality entirely: there is no target label in this unsupervised task."} Imputation, assembly and scaling are defined in section 10 and fitted within each training pipeline. Feature ordering is saved with the model manifest.',
         'A DataFrame ready to split; no model has been fitted.',
         'Adding quality, label, record_id or split_bucket to the feature vector creates leakage or meaningless predictors.',data_code)
    split_code=SPLIT
    if kind=='classification':
        split_code+='''\nfor name, frame in [("train", train), ("validation", validation), ("test", test)]:
    assert frame.select("label").distinct().count() == 2, f"{name} needs both classes for AUC"
train.groupBy("label").count().orderBy("label").show()
'''
    cell('9. Train/validation/test split and training EDA','Separate learning, settings selection and final assessment. Inspect training data only.',
         'Buckets 0–69 train, 70–84 validate, and 85–99 test. filter defines each subset; count triggers execution and assert catches empty splits. count(column) counts non-NULL training values; first retrieves one summary row. describe reports training distributions. Binary models check both classes in each subset so AUC is defined; only training balance is displayed. Hash splitting suits this independent teaching dataset, not temporal forecasting or known production batch groups.',
         'Three positive split counts and training summary statistics. Counts vary with dataset contents. Scaling statistics never use validation or test.',
         'If data is tiny or a class is missing, stop and design an appropriate split. Do not repeatedly resplit until test scores improve.',split_code)
    params=dict(fixed)
    if kind in ['regression','classification']: params.update(featuresCol='features',labelCol='label',predictionCol='prediction')
    elif kind=='clustering': params.update(featuresCol='features',predictionCol='prediction')
    else: params.update(inputCol='features',outputCol='pca_features')
    train_code=f'''from {module} import {klass}
ALGORITHM = {slug!r}
TASK = {kind!r}
CANDIDATE_SETTINGS = {configs!r}
FIXED_SETTINGS = {params!r}

def make_pipeline(settings):
    imputed_columns = [c + "_imputed" for c in feature_columns]
    imputer = Imputer(inputCols=feature_columns, outputCols=imputed_columns, strategy="median")
    assembler = VectorAssembler(inputCols=imputed_columns, outputCol="raw_features", handleInvalid="error")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    estimator = {klass}(**{{**FIXED_SETTINGS, **settings}})
    return Pipeline(stages=[imputer, assembler, scaler, estimator])

initial_model = make_pipeline(CANDIDATE_SETTINGS[0]).fit(train)
print("Fitted:", ALGORITHM)
'''
    cell('10. Model training','Fit an end-to-end preprocessing and modelling pipeline on training rows only.',
         f'Imputer(strategy="median") learns training medians. VectorAssembler creates an ordered numeric vector and fails on remaining invalid inputs. StandardScaler(withMean=True,withStd=True) learns training mean/std and makes dense standardized vectors; sensible for eleven features, not massive sparse text. Scaling is essential for distance-based models and retained consistently for all models here. Pipeline fits stages in order; fit(train) returns a PipelineModel. make_pipeline(settings) constructs a fresh pipeline per trial; dictionary expansion combines fixed and candidate settings. {param_text}',
         'A fitted pipeline message. No accuracy value is implied. Linear coefficients apply to the preprocessed representation, not raw chemical units.',
         'Unsupported ML classes or operations: switch to an appropriate supported environment. Null features: examine imputer inputs and training counts. If optimization fails to converge, inspect scales/data before raising maxIter.',train_code)
    pred_fields='"record_id", "label", "prediction"' if kind in ['regression','classification'] else '"record_id", "prediction"' if kind=='clustering' else '"record_id", "pca_features"'
    cell('11. Prediction','Apply the initial model to validation rows without learning anything new.',
         'transform executes the fitted imputer, assembler, scaler and model in order. select shows only identifiers and task outputs. limit/show bounds printed output. The PCA output is a vector of coordinates, not a class prediction.',
         'Five validation rows with '+('label and prediction; mismatches are normal.' if kind in ['regression','classification'] else 'cluster IDs; their numeric order has no quality meaning.' if kind=='clustering' else 'two PCA coordinates per row.'),
         'Use the full saved pipeline on raw input columns; do not independently refit a scaler on validation or new data.',f'validation_predictions = initial_model.transform(validation)\nvalidation_predictions.select({pred_fields}).show(5, truncate=False)')
    eval_code={'regression':REG_METRICS,'classification':CLS_METRICS,'clustering':CLUSTER_METRICS,'pca':PCA_METRICS}[kind]
    eval_code+='\nprint("Initial validation metrics:", evaluate(validation_predictions, initial_model))\n'
    cell('12. Model evaluation',METRIC_TEXT[kind],
         'evaluate returns a dictionary. RegressionEvaluator chooses each metric by metricName. Binary evaluation uses when/sum to aggregate TP/TN/FP/FN; first collects only this summary. BinaryClassificationEvaluator uses rawPrediction scores for threshold-independent ranking, not hard classes. ClusteringEvaluator uses standardized features and assignments. PCA uses vector_to_array and higher-order aggregate to sum squared coordinates per row; avg reduces reconstruction error, greatest clips tiny negative roundoff. explainedVariance comes from training PCA, and is labelled accordingly. selection_metric and maximize define model selection direction.',
         'Actual validation metrics printed by your run. No fabricated expected model scores are supplied. Read units and class definitions before comparing results.',
         'AUC requires both classes. Silhouette requires nontrivial groups. A constant regression target makes R² uninformative. Do not compare regression RMSE with classification F1.',eval_code)
    if kind=='pca':
        selection='''COMPLEXITY_PENALTY = 0.02  # Chosen before examining validation/test results.
def selection_value(trial):
    return trial["metrics"]["reconstruction_mse"] + COMPLEXITY_PENALTY * trial["settings"]["k"]
best_trial = min(trials, key=selection_value)'''
    else:
        selection='''best_trial = (max if maximize else min)(trials, key=lambda trial: trial["metrics"][selection_metric])'''
    tune_code='''trials = []
for index, settings in enumerate(CANDIDATE_SETTINGS):
    fitted = initial_model if index == 0 else make_pipeline(settings).fit(train)
    scores = evaluate(fitted.transform(validation), fitted)
    assert all(math.isfinite(float(value)) for value in scores.values()), "Invalid metric"
    trials.append({"settings": settings, "metrics": scores, "model": fitted})
    print("Candidate:", settings, scores)
'''+selection+'''
best_model = best_trial["model"]
print("Selected:", best_trial["settings"])
# Keep the selected train-fitted pipeline; no refit on test or validation.
test_predictions = best_model.transform(test)
test_metrics = evaluate(test_predictions, best_model)
print("Final test metrics:", test_metrics)
'''
    if kind=='regression': tune_code+='''train_mean = train.agg(F.avg("label").alias("mean")).first()["mean"]
baseline_metrics = evaluate(test.withColumn("prediction", F.lit(train_mean)))
print("Training-mean baseline test metrics:", baseline_metrics)
'''
    elif kind=='classification': tune_code+='''majority = train.groupBy("label").count().orderBy(F.desc("count"), F.asc("label")).first()["label"]
baseline_accuracy = test.select(F.avg((F.col("label") == majority).cast("double")).alias("accuracy")).first()["accuracy"]
baseline_metrics = {"accuracy": baseline_accuracy}
print("Training-majority baseline test accuracy:", baseline_accuracy)
test_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()
'''
    elif kind=='clustering': tune_code+='''baseline_metrics = {}
test_predictions.groupBy("prediction").agg(F.count("*").alias("rows"),
    *[F.avg(c).alias("mean_" + c) for c in feature_columns]).orderBy("prediction").show(truncate=False)
'''
    else: tune_code+='''baseline_metrics = {}
print("Training explained variance ratios:", best_model.stages[-1].explainedVariance)
'''
    cell('13. Hyperparameter tuning and final test','Compare a small, bounded grid using validation only, then evaluate the chosen model once on the test set.',
         'trials stores settings, validation metrics and pipelines. The first fitted model is reused, avoiding duplicate work. Each subsequent trial fits fresh preprocessing on train. min/max chooses the validation criterion; isfinite rejects invalid metrics. The model is not refitted on combined data in this teaching notebook. '+('For PCA, selection minimizes reconstruction_mse + 0.02*k: an explicit illustrative complexity tradeoff, not a universal best practice.' if kind=='pca' else '')+' Regression compares a training-mean baseline; classification compares training-majority accuracy and prints the confusion counts. K-Means prints per-cluster original-unit feature means. PCA prints training component variance ratios.',
         'Two candidate results, selected settings, and held-out metrics. A candidate can beat the baseline or fail; report the observed result honestly. Cluster feature means describe groups; do not rename them high/low quality.',
         'Never tune parameters or thresholds using test results. Comparing algorithms also requires validation-based selection; after choosing among notebooks, use a fresh holdout for an unbiased final project estimate.',tune_code)
    tracking='''mlflow.set_tracking_uri("databricks")
user_name = spark.sql("SELECT current_user() AS u").first()["u"]
experiment_path = f"/Users/{user_name}/databricks_datasets_ml_{TASK}"
mlflow.set_experiment(experiment_path)
for trial in trials:
    with mlflow.start_run(run_name=ALGORITHM + "_candidate"):
        mlflow.log_params({**FIXED_SETTINGS, **trial["settings"], "algorithm": ALGORITHM,
            "source_path": source_path, "split": "feature_hash_70_15_15", "spark_version": spark.version})
        mlflow.log_metrics({"validation_" + k: float(v) for k, v in trial["metrics"].items()})
with mlflow.start_run(run_name=ALGORITHM + "_selected") as run:
    mlflow.log_params({**FIXED_SETTINGS, **best_trial["settings"], "algorithm": ALGORITHM,
        "source_path": source_path, "split": "feature_hash_70_15_15", "spark_version": spark.version})
    mlflow.log_metrics({"validation_" + k: float(v) for k, v in best_trial["metrics"].items()})
    mlflow.log_metrics({"test_" + k: float(v) for k, v in test_metrics.items()})
    mlflow.log_metrics({"baseline_test_" + k: float(v) for k, v in baseline_metrics.items()})
    mlflow.log_dict({"features_in_order": feature_columns, "label_rule":
        "quality >=7" if TASK == "classification" else "quality" if TASK == "regression" else "none"},
        "feature_contract.json")
    selected_run_id = run.info.run_id
print("Experiment:", experiment_path)
print("Run:", selected_run_id)
'''
    if kind=='pca': tracking+='''with mlflow.start_run(run_id=selected_run_id):
    mlflow.log_param("complexity_penalty_per_component", COMPLEXITY_PENALTY)
'''
    cell('14. MLflow experiment tracking','Record comparable trials, their configuration, feature contract and final result.',
         'set_tracking_uri selects workspace tracking. current_user supplies your home experiment path. set_experiment creates/selects it. start_run creates one run per trial and a selected-model run; the with block closes each run. log_params records settings, log_metrics numeric results and log_dict a JSON artifact. The selected run includes validation results so model-family comparison need not use test scores. run.info.run_id is lineage used in outputs. Reopening that run later adds model paths. These are tracking records, not registry entries.',
         'Three runs per notebook. Compare algorithms within the same task by validation metrics in the shared task experiment. Exact run IDs are environment-specific.',
         'Permission denied: choose a writable experiment directory. Already-active run: finish your unrelated run explicitly. No package: add MLflow through the environment; do not overwrite built-in PySpark.',tracking)
    cell('15. Model saving and loading','Persist the entire preprocessing/model pipeline and a Delta source snapshot for later use.',
         'current_catalog/current_schema locate your namespace; change CATALOG/SCHEMA before writes if it is read-only. quote_identifier escapes SQL identifiers. CREATE VOLUME IF NOT EXISTS provisions a managed model volume if permitted. uuid4 gives each run an independent directory and table names. data.write.format("delta").mode("errorifexists") avoids overwrites. best_model.write().save stores native Spark PipelineModel stages; PipelineModel.load reloads them. The manifest logs the exact path and feature order to MLflow. In a new session, use the printed absolute model path. Only trusted pipeline artifacts should be loaded.',
         'A managed Delta input snapshot, a model directory and a reloaded PipelineModel. This is native Spark model persistence, not mlflow.spark.log_model and not Unity Catalog model registration.',
         'Use a writable non-samples namespace. Insufficient volume permissions require an existing writable volume or admin help. If PipelineModel persistence is unsupported on the selected serverless environment, use a supported environment: saving only coefficients cannot replace these pipelines.',SAVE)
    batch='''batch_input = test.select("record_id", *feature_columns)
# Demonstration: held-out rows stand in for a new incoming batch; labels are excluded.
# For real inference replace batch_input with a table having this feature contract.
normalized_batch = batch_input.select("record_id", *[finite_or_null(c).alias(c) for c in feature_columns])
scored = loaded_model.transform(normalized_batch)
'''
    if kind=='classification': batch+='''output = scored.select("record_id", "prediction", vector_to_array("probability")[1].alias("positive_probability"))
'''
    elif kind=='pca': batch+='''output = scored.select("record_id", vector_to_array("pca_features").alias("components"))
'''
    else: batch+='''output = scored.select("record_id", "prediction")
'''
    batch+='''output = output.withColumn("mlflow_run_id", F.lit(selected_run_id)).withColumn("scored_at", F.current_timestamp())
output.write.format("delta").mode("errorifexists").saveAsTable(prediction_table)
spark.table(prediction_table).show(5, truncate=False)
print("Prediction table:", prediction_table)
'''
    cell('16. Batch prediction','Apply the reloaded pipeline to a label-free batch and write queryable outputs.',
         'select removes quality/label and split metadata. finite_or_null applies the same deterministic cleanup to incoming features. transform applies saved training medians, scaler and model. vector_to_array converts ML vectors to standard arrays for Delta output; classification extracts positive-class probability at index 1, PCA stores component arrays. No vector UDT is written to Delta. withColumn records run ID and timestamp. saveAsTable persists scalar/array columns for SQL consumers.',
         'Five persisted output rows and a unique Delta table name. This demo reuses test features to show inference mechanics; it does not claim they are genuinely new observations.',
         'Missing feature names or different units break the contract. All-missing records may technically score after imputation but should be quarantined under a business rule. For production retries, use stable source IDs and model versions with idempotent MERGE logic.',batch)
    md(f'''## 17. Production considerations
- Preserve the full preprocessing pipeline, feature order, source path/snapshot, code commit, environment and training cutoff. This exercise snapshots data after training for later audit; production should train from an immutable, versioned snapshot created before training.
- Treat sensory quality scores as noisy, subjective labels. The >=7 binary cutoff is illustrative. New regions, vintages and laboratories can shift distributions; this dataset alone does not validate a production quality-control system.
- Split by time/producer/batch when that metadata exists. Identical-feature grouping helps duplicate leakage but cannot discover unknown producer groups.
- Compare with earlier models on identical validation splits. Once you choose a model family, reserve a fresh final holdout; repeatedly consulting the same test set biases evaluation.
- Monitor missing values and feature distributions. For supervised models, join delayed true outcomes and monitor task metrics; for clusters, monitor sizes/stability; for PCA, monitor reconstruction error.
- Retrain on a schedule or validated degradation, evaluate a candidate, retain rollback models, and promote only after quality gates.
- Native Spark pipeline persistence is demonstrated. Registering models additionally needs a supported MLflow model flavor, input/output signature and UC permissions. The JSON manifest is not a serving package. Batch jobs do not need a serving endpoint.
- Separate ingestion, training, validation and scoring job tasks. Move reusable functions into modules after understanding the standalone notebook; test preprocessing contracts, split isolation, metrics and reload parity with pytest. CI should lint/compile and run a small integration job before promotion.
- Free Edition is appropriate for learning, with quota and capability limits; production deployment needs a suitable workspace and operational controls. Small grids and tree counts here conserve resources.

### Three practical exercises
1. Explain `{list(configs[0])[0]}` and change only its validation candidate values. Compare validation results before reading test results.
2. Introduce one missing feature value in a training copy. Explain what the training-fitted median imputer does to that value and to a missing inference value.
3. Inspect the saved output table with SQL and trace a prediction/component/cluster back to its MLflow run and model path.

### Five interview questions with answers
1. **What does this algorithm do?** {intro}
2. **Which setting controls the learning behaviour?** {param_text}
3. **Why fit preprocessing on training only?** Statistics learned from validation/test would leak information into model selection or final assessment.
4. **How do you evaluate this task?** {METRIC_TEXT[kind]}
5. **How does it compare with previous methods?** {comparison}

### Troubleshooting scenario
**Symptom:** validation performance looked strong, but loading the saved model gives very different outputs. **Likely cause:** only the estimator was saved while imputation, feature ordering or scaling was rebuilt on new records. **Diagnosis:** compare feature names/order, units, model version and preprocessing parameters on identical input rows. **Fix:** save/load the entire PipelineModel as in section 15 and compare outputs within numerical tolerance.

### Mini assignment
Run this notebook in your workspace. Submit the source path, raw/clean/split counts, two validation trial results, chosen settings, final metrics, experiment run ID and model path. Explain one limitation and compare with an earlier notebook **within the same task**, without choosing based on test results.

### Short quiz
1. Which data should choose hyperparameters: training, validation or test?
2. Can quality be included in the eleven-feature input vector for this task?
3. Does a saved final estimator alone preserve a separately fitted scaler?

Answers: 1. Validation. 2. No; it is the supervised outcome or intentionally excluded from the unsupervised task. 3. No; save the full pipeline.

Pause here. You can create all notebooks now while studying them one at a time; do not move to the next lesson until you are ready.
''')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':4}
    (OUT/(slug+'.ipynb')).write_text(json.dumps(nb,indent=2)+'\n')
print(f'Generated {len(SPECS)} standalone notebooks in {OUT}')
