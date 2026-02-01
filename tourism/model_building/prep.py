
from sklearn.model_selection import train_test_split

# Define constants for the dataset and output paths
DATASET_PATH = "hf://datasets/Vvaannii33/Tourism-Package-Creation/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

# Define the target variable for the classification task
target = 'ProdTaken'

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

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]


for file_path in files:
  path_or_fileobj = file_path,
  path_in_repo = file_path.split("/")(-1),
  repo_id = "Vvaannii33/Tourism-Package-Creation",
  repo_type = "dataset"

