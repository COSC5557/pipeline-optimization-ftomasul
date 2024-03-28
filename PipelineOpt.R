# Finn Tomasula Martin
# COSC-4557
# Hyperparameter Optimization
# This file contains the code for the hyperparameter optimization exercise

# Clear environment
rm(list = ls())
while (!is.null(dev.list())) dev.off()

# Load libraries
library(caret)
library(randomForest)
library(e1071)
library(rBayesianOptimization)