# Machine Learning in Databricks

Hands-on learning notebooks using Python, **PySpark / Spark ML**, MLflow and Delta Lake. No pandas, RDD API, DBFS mounts or DataFrame cache calls are required.

## Start here

1. Open [Lesson 1: ML fundamentals](notebooks/Lesson_01_ML_Fundamentals_Databricks.ipynb) if you want the small synthetic introduction first.
2. For real sample data, start with [Linear Regression](notebooks/algorithms/02_linear_regression.ipynb).
3. Import the `.ipynb` into Databricks and attach a recent notebook environment with supported Spark ML APIs. Do **not** install PySpark into Databricks yourself.
4. Run cells sequentially. If MLflow is absent, add it through notebook Environment dependencies, restart and rerun.
5. Before persistence, set `CATALOG` and `SCHEMA` to a writable Unity Catalog namespace. The notebook creates a managed volume named `ml_learning_models` if permitted. An existing writable volume can be substituted by changing `VOLUME`.
6. Study one notebook at a time. Generating the complete set does not require advancing lessons before you are ready.

## Algorithms and dataset

All 12 algorithm notebooks use the same real Databricks sample CSV:

`/databricks-datasets/wine-quality/winequality-red.csv`

Databricks documents this UCI Wine Quality dataset in its [ML getting-started guide](https://docs.databricks.com/aws/en/getting-started/ml-get-started). It contains 11 chemical measurements plus a sensory quality score, with **semicolon-separated** fields. The original red-wine file commonly has 1,599 rows before deduplication; each notebook reports the count it actually reads.

| Notebook | Task | Validation selection |
|---|---|---|
| [02 Linear Regression](notebooks/algorithms/02_linear_regression.ipynb) | Predict quality score | Lowest RMSE |
| [03 Ridge Regression](notebooks/algorithms/03_ridge_regression.ipynb) | Linear prediction with L2 penalty | Lowest RMSE |
| [04 Lasso Regression](notebooks/algorithms/04_lasso_regression.ipynb) | Linear prediction with L1 penalty | Lowest RMSE |
| [05 Decision Tree Regression](notebooks/algorithms/05_decision_tree_regression.ipynb) | Nonlinear score prediction | Lowest RMSE |
| [06 Random Forest Regression](notebooks/algorithms/06_random_forest_regression.ipynb) | Ensemble score prediction | Lowest RMSE |
| [07 Gradient Boosted Tree Regression](notebooks/algorithms/07_gradient_boosted_tree_regression.ipynb) | Sequentially corrected score prediction | Lowest RMSE |
| [08 Logistic Regression](notebooks/algorithms/08_logistic_regression.ipynb) | Predict quality >=7 | Highest PR-AUC |
| [09 Decision Tree Classification](notebooks/algorithms/09_decision_tree_classification.ipynb) | Predict quality >=7 | Highest PR-AUC |
| [10 Random Forest Classification](notebooks/algorithms/10_random_forest_classification.ipynb) | Predict quality >=7 | Highest PR-AUC |
| [11 Gradient Boosted Tree Classification](notebooks/algorithms/11_gradient_boosted_tree_classification.ipynb) | Predict quality >=7 | Highest PR-AUC |
| [12 K-Means](notebooks/algorithms/12_kmeans_clustering.ipynb) | Group chemical profiles, no label | Highest held-out silhouette |
| [13 PCA](notebooks/algorithms/13_pca_dimension_reduction.ipynb) | Compress chemical features, no label | Reconstruction MSE + explicit component penalty |

The >=7 cutoff is a teaching definition, not a commercial quality standard. PCA is dimensionality reduction, not a prediction classifier. K-Means cluster IDs have no inherent quality order. Ridge/Lasso use Spark LinearRegression with different penalty settings. Spark GBTClassifier is binary; these GBT notebooks are not XGBoost.

## Included in each algorithm notebook

All 17 requested sections: business problem; introduction; use case; dataset; loading; EDA; cleaning; feature engineering; splitting; training; prediction; evaluation; tuning; MLflow; saving/loading; batch inference; production considerations.

Every code cell has separate Markdown covering purpose, functions/parameters, expected output interpretation and common errors. Each notebook ends with three exercises, five interview questions with answers, a troubleshooting scenario, a mini assignment and a short quiz.

### Evaluation and leakage controls

- Remove exact full-record duplicates; hash **feature profiles only** into approximate 70/15/15 train/validation/test buckets. Identical inputs stay in the same split, even if labels differ.
- Use the same split across algorithms. Do training EDA only after splitting.
- Fit median imputation and standardization inside the training pipeline. All candidate models use only training-fitted preprocessing.
- Two candidates per notebook keep the initial grid small. Select using validation only, then evaluate test results.
- Classification metrics use explicit positive-class precision/recall/F1 and confusion counts, plus ROC-AUC and PR-AUC. They do not mislabel weighted multiclass F1 as positive-class F1.
- Regression includes a training-mean baseline. Classification includes training-majority baseline accuracy.
- PCA explained variance is explicitly labelled **training** explained variance; held-out reconstruction error is separate. A declared component penalty illustrates compression/error tradeoffs.
- Compare model families using validation metrics. If you repeatedly inspect the same test results across models, reserve a fresh holdout for final project assessment.
- Unknown wine producer/batch relationships cannot be controlled because this CSV lacks that metadata. These splits do not establish production generalization.

### Persistence and tracking

- Candidate and selected-model MLflow runs include parameters, dataset path, validation metrics, test metrics and a feature contract.
- The **entire native Spark PipelineModel** is saved in a Unity Catalog volume and reloaded for inference. Imputer, feature order, scaling and estimator stay together.
- MLflow records a manifest pointing to that pipeline. This is **not** MLflow model-flavor logging, registration or a serving deployment.
- Unique managed Delta tables retain the source snapshot and scalar/array prediction outputs. Existing tables are not overwritten.
- Demo inference uses held-out feature rows with labels removed. Replace that input with genuinely incoming data in a production scoring job.
- Registration/serving require compatible model packaging, signatures and workspace permissions; those operations are not silently attempted.

## Free Edition and dataset access

Free Edition uses serverless compute with quotas and restrictions. API and storage support depend on the selected environment. These notebooks avoid RDDs, SparkContext, cache/persist and mounts, but they require support for the shown Spark ML estimators and PipelineModel persistence.

If `/databricks-datasets` is unavailable:

1. Inspect the sample volume `/Volumes/samples/databricks/datasets/` in your workspace for an accessible copy. Its contents are environment-dependent; these notebooks do not assume the legacy directory structure is mirrored there.
2. Alternatively, upload the **same** UCI red-wine CSV into a Unity Catalog volume you can read.
3. Set `DATA_PATH_OVERRIDE` to the exact accessible file path. Do not silently replace the dataset with synthetic data.

If an ML operation is unsupported, choose a supported environment/compute configuration. A Python import succeeding does not prove remote training or persistence is supported. There is no universal fallback implementation in these algorithm notebooks. Training can run before storage cells; successful durable saving requires table/volume permissions.

Useful official references:
- [Databricks sample datasets](https://docs.databricks.com/aws/en/discover/databricks-datasets)
- [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)
- [Serverless limitations](https://docs.databricks.com/aws/en/compute/serverless/limitations)
- [Spark classification/regression](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [Spark clustering](https://spark.apache.org/docs/latest/ml-clustering.html)
- [Spark features/PCA](https://spark.apache.org/docs/latest/ml-features.html)

## Reproducible generation and validation

`scripts/build_algorithm_notebooks.py` generates the twelve standalone notebooks from common explanations and explicit per-algorithm settings. It uses the Python standard library only. Run from any directory:

```bash
python scripts/build_algorithm_notebooks.py
python scripts/validate_notebooks.py
```

The validator parses notebook JSON, checks Python syntax, verifies the teaching sections and verifies that new algorithm notebooks contain no executed/fabricated outputs. **This is static validation, not a Databricks training run.** Dataset access, fitting, MLflow operations and pipeline save/load must be verified in your own workspace. Numeric metric examples in prose explain arithmetic; actual model scores are generated only when you run the notebook.

Later lessons can add Naive Bayes, SVM, XGBoost/LightGBM, forecasting, recommendations, neural networks and the end-to-end customer-churn project with task-appropriate datasets. This commit contains the twelve algorithms listed above.
