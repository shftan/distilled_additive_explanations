# Python pseudocode using sklearn-inspired notation.

#########################################################################
### 1) Load data and split it into train, val, and test.
#########################################################################

# In our experiments we first do 15% test, 85% trainval. 
# The 85% trainval is later split into 15% val and 70% train.
percentage_test = 0.15
percentage_val = 0.15
percentage_train = 0.70

# Load data and split
all_data = load_data()
test, trainval = train_test_split(
    all_data,
    test_size=percentage_test,
    train_size=(percentage_val + percentage_train),
    random_state=0,
    shuffle=True
)

# Is this a classification task, or a regression task? 
# This will affect metrics and losses
# Metric: AUC for classification problems and RMSE for regression problems
# Loss: cross-entropy for classification problems and RMSE for regression problems
is_classification = all_data.is_classification
metric = AUC if is_classification else RMSE
loss = cross_entropy if is_classification else RMSE

#########################################################################
### 2) Train black-box model to interpret. 
### Use cross validation using different train/val splits.
#########################################################################

# num_layers can be adjusted manually if aiming at a given depth.
num_layers = 2

# Train black-box model
model, model_parameters = train_crossvalidated_model(
    estimator=MLP,
    loss=loss,
    data=trainval.data,
    labels=trainval.labels,
    num_folds=5,
    val_size=percentage_val / (percentage_train + percentage_val),
    estimator_cv_parameters={
        "max_epochs":200,
        "early_stopping": True,
        "num_layers": num_layers,
        "num_hidden_units": [8, 16, 32, 64, 128, 256],
        "p_dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [1e-2, 1e-3, 1e-4, 1e-5] + 5 * [ 1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] + 5 * [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        "batch_size": [8, 16, 32, 64, 128, 256]
    },    
    search_strategy="random_search",
    metric=metric,
    num_tries=100
)

# Measure performance of model:
# AUC for classification problems and RMSE for regression problems
performance = evaluate_model(
    model=model,
    data=test.data, 
    labels=test.labels,
    metric=metric
)
print(f"Performance of black box model: {performance:.2f}")

#########################################################################
### 3) Train and evaluate student model
#########################################################################

# First, train a student model with optimal parameters using cross-validation. 
# We can train, for example, a student based on Bagged Additive Boosted Trees (SAT in the paper), 
# sometimes also referred to as Explainable Boosting Machine (EBM).
# For this we recommend the public implementation of InterpretML: https://interpret.ml/docs/ebm.html. 
# Note that in the paper we used a different, proprietary implementation, but results should be equivalent.
# These models have a lot less parameters and these parameters usually have a smaller impact, so we reduce the number of tries.
# The goal here is to obtain the optimal parameters, we will retrain more student models later.
trainval_predictions = model.predict(trainval.data)
student, student_parameters = train_crossvalidated_model(
    estimator=ExplainableBoostingRegressor,
    loss=RMSE,
    data=trainval.data,
    labels=trainval_predictions,
    num_folds=5,
    val_size=percentage_val / (percentage_train + percentage_val),
    estimator_cv_parameters={
        "interactions": 0,
        "max_rounds": [100, 1000, 5000],
        "max_bins": [128, 256, 512],
    },
    search_strategy="random_search",
    metric=metric,
    num_tries=10
)

# Train different student models using different train/val partitions and average results.
results = []
for seed in range(5):
    # Split data according to seed
    val, train = train_test_split(
        trainval, 
        test_size=percentage_val / (percentage_train + percentage_val), 
        train_size=percentage_train / (percentage_train + percentage_val), 
        random_state=seed, 
        shuffle=True
    )

    # Train the student using the train predictions as target and the best student parameters
    student = ExplainableBoostingRegressor(**student_parameters)
    student.fit(
        data=train.data, 
        labels=model.predict(train.data),
    )

    # Evaluate and add to results
    student_performance = evaluate_model(
        model=student,
        data=test.data, 
        labels=test.labels,
        metric=metric
    )
    results.append(student_performance)

# Report average performance and std
print(f"{np.mean(results):.3f} +- {np.std(results):.3f}") 


#########################################################################
### 5) Display student
#########################################################################

# For the ExplainableBoostingRegressor provided by InterpretML, an explain_global
# method exists that does all the work. We recommend investigating the 
# code inside explain_global to see how this can be applied to other student models.
show(student.explain_global())
