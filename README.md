# Pipeline Optimization

For this exercise, we will have a look at Hyperparameter Optimization of ML
pipelines -- instead of optimizing a single component, we want to optimize the
entire ML approach, including any preprocessing etc. The end result (i.e. the
predictive performance) is not important; how you get there is.

Your deliverable will be a report, written in a style that it
would be suitable for inclusion in an academic paper as the "Experimental
Setup" section or similar. If unsure, check an academic paper of your choice,
for example [this one](https://www.eecs.uwyo.edu/~larsko/papers/pulatov_opening_2022-1.pdf). The
level of detail should be higher than in a typical academic paper though. Your
report should be at most five pages, including references and figures but
excluding appendices. It should have the following structure:
- Introduction: What problem are you solving, how are you going to solve it.
- Dataset Description: Describe the data you're using, e.g. how many features and observations, what are you predicting, any missing values, etc.
- Experimental Setup: What specifically are you doing to solve the
        problem, i.e. what programming languages and libraries, how are you
        processing the data, what preprocessing steps, machine learning
        algorithms, and other components are you considering and what
        hyperparameters and value ranges, what measures you are using to
        evaluate them, what hyperparameter optimization method you chose, etc.
- Results: Description of what you observed, including plots. Compare
  performance before and after tuning, and show the best configuration.
- Code: Add the code you've used as a separate file.

Your report must contain enough detail to reproduce what you did without the
code. If in doubt, include more detail.

There is no required format for the report. You could, for example, use an
iPython notebook.

## Data and Setup

Have a look at two different datasets. Ideally, they should have different
characteristics so that the results of your analysis will be different. You
could use, for example, the [Wine Quality dataset](https://archive-beta.ics.uci.edu/dataset/186/wine+quality)
(choose the type of wine of your preference) and the
[Primary Tumor dataset](https://www.openml.org/search?type=data&sort=runs&id=1003&status=active).
You may also use datasets of your choice, for example ones that are relevant to
your research.

Choose a small number of different machine learning algorithms and
hyperparameters, along with sensible value ranges, and additional ML pipeline
components. This could include, for example, a feature imputation step, a
feature recoding step, and an ensembling or stacking step in addition to the
learning algorithm. If those components have hyperparameters, you must choose
ranges and tune them as well. In addition, your pipeline should allow the
hyperparameter optimization to turn individual components on and off, e.g. use a
one-hot-encoding or not. Your hyperparameter search space should be so large
that you cannot simply run a grid search.

You can use implementations of AutoML systems (e.g. auto-sklearn), scientific
papers, or the documentation of the library you are using to determine the
hyperparameters to tune and the value ranges. Note that there is not only a
single way to do this, but define a reasonable space (e.g. don't include whether
to turn on debug output, or random forests with 1,000,000 trees, or tune the
loss function). Your ML pipeline needs to be reasonably complex, i.e. at least
three components.

Determine the best ML pipeline. Make sure to optimize the entire ML pipeline,
not individual parts individually. Choose a suitable hyperparameter optimizer;
you could also use several and e.g. compare the results achieved by random
search and Bayesian optimization. Make sure that the way you evaluate this
avoids bias and overfitting. You could use statistical tests to make this
determination.

## Submission

Add your report and code to this repository. Bonus points if you can set up a
Github action to automatically run the code and generate the report!
