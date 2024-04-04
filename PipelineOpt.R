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
encode = po("encode", method = to_tune(c("one-hot", "treatment", "helmert", "poly", "sum")))

# Define base learners
base_kknn = as_learner(impute %>>% lrn("classif.kknn", id = "base_kknn"))
base_rpart = as_learner(impute %>>% lrn("classif.rpart", id = "base_rpart"))
base_ranger = as_learner(impute %>>% lrn("classif.ranger", id = "base_ranger"))

# Define models to be optimized
kknn = lrn("classif.kknn", k = to_tune(1, 20), distance = to_tune(1, 20), predict_type = "prob", id = "kknn")
rpart = lrn("classif.rpart", maxdepth = to_tune(1, 30), minbucket = to_tune(1, 50), id = "rpart")
ranger = lrn("classif.ranger", num.trees = to_tune(100, 1000), mtry = to_tune(1, 8), min.node.size = to_tune(1, 10), predict_type = "prob", id = "ranger")

# Define full pipeline for each model
kknn_stack = as_learner(impute %>>% encode %>>% scale %>>%
  gunion(list(
    po("learner_cv", lrn("classif.kknn", predict_type = "prob")),
    po("learner_cv", lrn("classif.ranger", predict_type = "prob")))) %>>%
  po("featureunion") %>>% 
    kknn, id = "opt_kknn")
rpart_stack = as_learner(impute %>>% encode %>>% scale %>>%
  gunion(list(
   po("learner_cv", lrn("classif.kknn", predict_type = "prob")),
   po("learner_cv", lrn("classif.ranger", predict_type = "prob")))) %>>%
  po("featureunion") %>>%  
    rpart, id = "opt_rpart")
ranger_stack = as_learner(impute %>>% encode %>>% scale %>>%
  gunion(list(
    po("learner_cv", lrn("classif.kknn", predict_type = "prob")),
    po("learner_cv", lrn("classif.ranger", predict_type = "prob")))) %>>%
  po("featureunion") %>>% 
    ranger, id = "opt_ranger")

pipelines = list(kknn_stack, rpart_stack, ranger_stack)

# Define a tuner with Bayesian optimization
optimized_pipelines = lapply(pipelines, function(pipeline) {
  auto_tuner(
    tuner = tnr("mbo"),
    learner = pipeline,
    resampling = rsmp("cv", folds = 10),
    measure = msr("classif.ce"),
    terminator = trm("evals", n_evals = 100)
  )
})

learners = c(base_kknn, base_rpart, base_ranger, optimized_pipelines)

# Benchmark all models to get the results
resamp = rsmp("holdout")
design = benchmark_grid(tasks, learners, resamp)
results = benchmark(design, store_models = TRUE)

# Optimal configuration for each model
kknn_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, kknn.k, kknn.distance, classif.ce, task_id)]
kknn_result = kknn_result[!is.na(kknn.k) & !is.na(kknn.distance)]
kknn_wine_opt = kknn_result[which.min(classif.ce) & task_id == "wine"]
kknn_tumor_opt = kknn_result[which.min(classif.ce) & task_id == "tumor"]

rpart_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, rpart.maxdepth, rpart.minbucket, classif.ce, task_id)]
rpart_result = rpart_result[!is.na(rpart.maxdepth) & !is.na(rpart.minbucket)]
rpart_wine_opt = rpart_result[which.min(classif.ce) & task_id == "wine"]
rpart_tumor_opt = rpart_result[which.min(classif.ce) & task_id == "tumor"]

ranger_result = extract_inner_tuning_results(results)[,.(scale.robust, encode.method, ranger.num.trees, ranger.mtry, ranger.min.node.size, classif.ce, task_id)]
ranger_result = ranger_result[!is.na(ranger.num.trees) & !is.na(ranger.num.trees)]
ranger_wine_opt = ranger_result[which.min(classif.ce) & task_id == "wine"]
ranger_tumor_opt = ranger_result[which.min(classif.ce) & task_id == "tumor"]

# Compare models and visualize the results
results$aggregate()[, .(task_id, learner_id, classif.ce)]
plot = autoplot(results, type = "boxplot")




