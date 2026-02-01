

#for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,recall_score

#for model serializtion
import joblib

# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError
from huggingface_hub import login,HfApi,create_repo


api= HfApi()

Xtrain_path = "hf://datasets/Vvaannii33/Tourism-Package-Creation/Xtrain.csv"
Xtest_path = "hf://datasets/Vvaannii33/Tourism-Package-Creation/Xtest.csv"
ytrain_path = "hf://datasets/Vvaannii33/Tourism-Package-Creation/ytrain.csv"
ytest_path = "hf://datasets/Vvaannii33/Tourism-Package-Creation/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# List of numerical features in the dataset
numeric_features = [
    'Age',                          #Age of the customer.
    'CityTier',                     #The city category based on development, population, and living standards 
    'DurationOfPitch',              #Duration of the sales pitch delivered to the customer.
    'NumberOfPersonVisiting',       #Total number of people accompanying the customer on the trip.
    'NumberOfFollowups',            #Total number of follow-ups by the salesperson after the sales pitch
    'PreferredPropertyStar',        #Preferred hotel rating by the customer.
    'NumberOfTrips',                #Average number of trips the customer takes annually
    'Passport'                      #Whether the customer holds a valid passport
    'PitchSatisfactionScore'        #Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar',                       #Whether the customer owns a car
    'NumberOfChildrenVisiting',     #Number of children below age 5 accompanying the customer.
    'MonthlyIncome'                 #Gross monthly income of the customer.

]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',                #The method by which the customer was contacted
    'Occupation',                   #Customer's occupation 
    'Gender',                       #Gender of the customer 
    'ProductPitched',               #The type of product pitched to the customer.
    'MaritalStatus'                 #Marital status of the customer
]

class_weight = ytrain.value_counts()[0]/ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps

preprocessor = make_column_transformer(
    (StandardScaler(),numeric_features),
    (OneHotEncoder(handle_unknown='ignore'),categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight,random_state=42)

# Define hyperparameter grid

param_grid = { 

              'xgbClassifier__n_estimators' : [50,75,100,125,150],
              'xgbClassifier__max_depth' : [2,3,4],
              'xgbClassifier__colsample_by_tree' : [0.4,0.5,0.6],
              'xgbClassifier__colsample_by_level' : [0.4,0.5,0.6],
              'xgbClassifier__learning_rate' : [0.01,0.05,0.1],
              'xgbClassifier_reg_lambda' : [0.4,0.5,0.6],

}

# Model pipeline
model_pipeline = make_pipeline(preprocessor,xgb_model)

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)


# Check the parameters of the best model
grid_search.best_params_

# Store the best model
best_model = grid_search.best_estimator_
best_model

# Set the classification threshold
classification_threshold = 0.45

# Make predictions on the training data
y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

# Make predictions on the test data
y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

# Generate a classification report to evaluate model performance on training set
print(classification_report(ytrain, y_pred_train))

# Generate a classification report to evaluate model performance on test set
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "tourism_package_creation.joblib")

# Upload to Hugging Face
repo_id = "Vvaannii33/Tourism-Package-Creation"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="tourism_package_creation.joblib",
    path_in_repo="tourism_package_creation.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)


