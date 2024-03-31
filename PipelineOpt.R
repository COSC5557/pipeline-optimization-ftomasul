# Finn Tomasula Martin
# COSC-4557
# Pipeline Optimization
# This file contains the code for the pipeline optimization exercise

# Clear environment
rm(list = ls())
while (!is.null(dev.list())) dev.off()

# Load libraries
library(mlr3verse)
library(kknn)
library(rpart)
library(ranger)
library(e1071)
library(xgboost)

# Initialize seed for reproducibility 
set.seed(123)

# Load in data
wine = read.csv("winequality-red.csv", sep=";")
tumor = read.csv("primary-tumor.csv")

# Modify features of type character
feature_types <- sapply(tumor, class)
character_features <- names(which(feature_types == "character"))
tumor$class <- as.factor(tumor$class)
for(feature in character_features) {
  tumor[[feature]] = as.factor(tumor[[feature]])
}

# Define tasks
wine_task = as_task_classif(wine, target = "quality")
tumor_task = as_task_classif(tumor, target = "class")
tasks = list(wine_task, tumor_task)

# Define preprocessing PipeOps
impute = po("imputemode")
scale = po("scale", robust = to_tune(c(TRUE, FALSE)))
base_encode = po("encode")
encode = po("encode", method = to_tune(c("one-hot", "treatment", "helmert", "poly", "sum")))

# Define base learners
base_kknn = as_learner(impute %>>% base_encode %>>% lrn("classif.kknn", id = "base_kknn"))
base_rpart = as_learner(impute %>>% base_encode %>>% lrn("classif.rpart", id = "base_rpart"))
base_ranger = as_learner(impute %>>% base_encode %>>% lrn("classif.ranger", id = "base_ranger"))
base_svm = as_learner(impute %>>% base_encode %>>% lrn("classif.svm", id = "base_svm"))
base_gboost = as_learner(impute %>>% base_encode %>>% lrn("classif.xgboost", id = "base_gboost"))

# Define pipeline for each model to be optimized
kknn = as_learner(impute %>>% scale %>>% encode %>>% lrn("classif.kknn", k = to_tune(1, 20), distance = to_tune(1, 20), id = "kknn"))
rpart = as_learner(impute %>>% scale %>>% encode %>>% lrn("classif.rpart", maxdepth = to_tune(1, 20), minbucket = to_tune(1, 20), id = "rpart"))
ranger = as_learner(impute %>>% scale %>>% encode %>>% lrn("classif.ranger", num.trees = to_tune(50, 200), mtry = to_tune(1, 10), min.node.size = to_tune(1, 10), id = "ranger"))
svm = as_learner(impute %>>% scale %>>% encode %>>% lrn("classif.svm", cost = to_tune(1e-1, 1e5), gamma = to_tune(1e-1, 1), id ="svm"))
gboost = as_learner(impute %>>% scale %>>% encode %>>% lrn("classif.xgboost", eta = to_tune(1e-4, 1, logscale = TRUE), max_depth = to_tune(1, 20), id = "gboost"))

# Define a model stack for each pipeline
kknn_stack = impute %>>% scale %>>% encode %>>%
  gunion(list(base_rpart, base_ranger, base_svm, base_gboost)) %>>%
  po("featureunion") %>>%
  lrn("classif.kknn", k = to_tune(1, 20), distance = to_tune(1, 20), id = "kknn")

pipelines = list(kknn, rpart, ranger, svm, gboost)

# Define a tuner with Bayesian optimization
optimized_pipelines = lapply(pipelines, function(pipeline) {
  auto_tuner(
    tuner = tnr("mbo"),
    learner = pipeline,
    resampling = rsmp("cv", folds = 10),
    measure = msr("classif.ce"),
    terminator = trm("evals", n_evals = 5)
  )
})

learners = c(base_kknn, base_rpart, base_ranger, base_svm, optimized_pipelines)

# Benchmark all models to get the results
cv = rsmp("cv", folds = 10)
design = benchmark_grid(tasks, learners, cv)
results = benchmark(design, store_models = TRUE)
results$aggregate()[, .(task_id, learner_id, classif.ce)]

# Optimal configuration for each model
kknn_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, kknn.k, kknn.distance, classif.ce)]
kknn_result = kknn_opt[!is.na(kknn.k) & !is.na(kknn.distance)]
kknn_opt = kknn_result[which.min(classif.ce)]

rpart_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, rpart.maxdepth, rpart.minbucket, classif.ce)]
rpart_result = rpart_results[!is.na(rpart.maxdepth) & !is.na(rpart.minbucket)]
rpart_opt = rpart_result[which.min(classif.ce)]

ranger_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, ranger.num.trees, ranger.mtry, ranger.min.node.size, classif.ce)]
ranger_result = ranger_result[!is.na(ranger.num.trees) & !is.na(ranger.mtry) & !is.na(ranger.num.trees)]
ranger_opt = ranger_result[which.min(classif.ce)]

svm_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, svm.cost, svm.gamma, classif.ce)]
svm_result = svm_result[!is.na(svm.cost) & !is.na(svm.gamma)]
svm_opt = svm_result[which.min(classif.ce)]

gboost_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, gboost.eta, gboost.max_depth, classif.ce)]
gboost_result = gboost_result[!is.na(gboost.eta) & !is.na(gboost.max_depth)]
gboost_opt = gboost_result[which.min(classif.ce)]

# Visualize the results
autoplot(results, measure = msr("classif.ce"))






