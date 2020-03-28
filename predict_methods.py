'''This script predicts temperature from three different machine learning models.
    The models are hosted on amazon s3 bucket
    Before starting, you should:
    1)If not done, upload your dataset to your s3 bucket
    2) Set the s3 bucket credentials (access key id and secret access key).
        *If you don't have these credentials, please follow the first part of this tutorial to get them https://realpython.com/python-boto3-aws-s3/
    3)Change the s3_bucket parameter to your bucket name
    4) Set your dataset name as hosted oin your bucket (change the input_name parameter)
'''

#Set the s3 credentials
aws_key_id='', #Change here, set your s3 access key id
aws_secret_key='', #Change here, set your s3 secret access key

#Set the bucket and input file name
s3_bucket = '' #Change here, set your bucket name
input_name = '' #Change here. Set your input file name that will be predicted (file needs to be hosted on s3 bucket)

#Don't change, these are the parameters asked by the assignment.
output_name = 'result.csv' #Output file for results after predicting
model1 = 'https://log8415-tp2-ml.s3.amazonaws.com/model1.pkl' #Model 1: SVM. Don't change
model2 = 'https://log8415-tp2-ml.s3.amazonaws.com/model2.pkl' #Model 2: RandomForestClassifier. Don't change
model3 = 'https://log8415-tp2-ml.s3.amazonaws.com/model3.pkl' #Model 3: DecisionTreeClassifier. Don't change

'''-------------------------------------------------------------------------'''
import pip
print('------Installing required libraries------')
def install(package_name):
  try:
      pipcode = pip.main(['install', package_name])
      if pipcode != 0:
          print("Unable to install " + package_name + " ; pipcode %d" % pipcode)
  except:
      print(package_name + " is already installed")
pkgs = ['wget', 'boto3', 'pandas', 'scikit-learn', 'pickle']
for pkg in pkgs:
  install(pkg)

import boto3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import sys
import wget

print('------Starting the script----------------')
#Initialising the connection to the s3 client.
try:
  print('Initialising the s3 bucket...', end ="   "),
  s3 = boto3.client(
      's3',
      aws_access_key_id=''.join(aws_key_id), #your s3 access key id
      aws_secret_access_key=''.join(aws_secret_key), #your s3 secret access key
  )
  print('DONE')
except Exception as e:
  print('Error! Did you set the s3 credentials correctly?')
  print(e)
  sys.exit('Please open the script and set the credentials. Exiting...')

#This function takes as input a panda Series (column) and categorize that column from verycold to veryhot
def categorizeTemp(column):
    temp_cat = column.apply(
    lambda x: 'verycold' if x <0 else (
        'cold' if 0<=x< 10 else (
            'moderate' if 10<=x<20 else (
                'hot' if 20<=x<30 else 'veryhot'))))
    return temp_cat

# Function for Encoding String values into numerical values
def encode(data, columns):
  for column in columns:
    encs = {}
    encs[column] = LabelEncoder()
    data[column] = encs[column].fit_transform(data[column])
  return encs

# unEncoding back numerical values into String values
def unencode(enc, data, columns):
  for column in columns:
    data[column] = enc.inverse_transform(data[column])

print("Reading input file...", end ="   "),
try:
  s3.download_file(s3_bucket, input_name,"data.csv")
  df = pd.read_csv('data.csv')
  print('DONE')
except:
  sys.exit('Error! Please specify your s3 bucket name (s3_bucket) and/or your input file name (input_name). Exiting... %tb')

print('Starting Data preprocessing...')
print('   +Removing unecessary columns No, PM2.5, PM10, SO2, NO2, CO and O3...', end ="   "),
try:
  cols = ['No','PM2.5','PM10','SO2','NO2','CO','O3']
  data = df.drop(cols,axis=1)
except:
  sys.exit('Error! The dataset should have the same columns as those given during the assignment. Exiting...')
print('DONE')

print('   +Dropping NaN values...'.ljust(10), end ="   ")
data = data.dropna()
print('DONE')

print('   +Categorizing temperature column TEMP from verycold to veryhot...', end ="   "),
data.TEMP = categorizeTemp(data.TEMP)
print('DONE')

print('   +Encoding String values into numerical values (wd, station and TEMP)', end ="   "),
encs = encode(data, ['wd', 'station', 'TEMP'])
print('DONE')

print('   +Applying feature scalling for data normalization...', end ="   ")
scaler = StandardScaler()
X_true = data.drop('TEMP', axis=1)
y_true = data['TEMP']
scaler.fit(X_true)
X_true = scaler.transform(X_true)
print('DONE')

print('Predicting with first model SVM...')
print('   +Downloadind the model from s3 bucket....', end='   ')
wget.download(model1, 'model1.pkl')
with open('model1.pkl', 'rb') as f:
    pkl_model1 = pickle.load(f)

# Calculate the accuracy score and predict target values
y_pred_model1 = pkl_model1.predict(X_true)
score1 = accuracy_score(y_true, y_pred_model1)
print('DONE')
print('   +\33[32m' + 'Accuracy score of model 1 : {0:.2f} %'.format(100 * score1) + '\33[0m')

print('Predicting with second model RandomForestClassifier...')
print('   +Downloadind the model from s3 bucket....', end='   ')
wget.download(model2, 'model2.pkl')
with open('model2.pkl', 'rb') as f:
    pkl_model2 = pickle.load(f)

# Calculate the accuracy score and predict target values
y_pred_model2 = pkl_model2.predict(X_true)
score2 = accuracy_score(y_true, y_pred_model2)
print('DONE')
print('   +\33[32m' + 'Accuracy score of model 2 : {0:.2f} %'.format(100 * score2) + '\33[0m')

print('Predicting with third model DecisionTreeClassifier...')
print('   +Downloadind the model from s3 bucket....', end='   ')
wget.download(model3, 'model3.pkl')
with open('model3.pkl', 'rb') as f:
    pkl_model3 = pickle.load(f)

# Calculate the accuracy score and predict target values
y_pred_model3 = pkl_model3.predict(X_true)
score3 = accuracy_score(y_true, y_pred_model3)
print('DONE')
print('   +\33[32m' + 'Accuracy score of model 3 : {0:.2f} %'.format(100 * score3) + '\33[0m')
print('Saving the predicted results to output file...', end='   ')

df_res = pd.DataFrame({'TEMP_true': data.TEMP, 'TEMP_model1': y_pred_model1, 'TEMP_model2': y_pred_model2, 'TEMP_model3': y_pred_model3})
unencode(encs['TEMP'], df_res, ['TEMP_true', 'TEMP_model1', 'TEMP_model2', 'TEMP_model3'])
df_res.to_csv (r'result.csv', index = False, header=True)
print('DONE')

print('Uploading result to s3 bucket...', end='   ')
with open(output_name, "rb") as f:
  s3.upload_fileobj(f, s3_bucket, output_name,
                    ExtraArgs={'ACL': 'public-read'}
                    )
print('DONE')
print('------------------Program completed successfully! showing predicted results--------------------')
print('Description of the columns of the output file:')
print('   +TEMP_true: True temperature from the dataset')
print('   +TEMP_model1: Predicted temperature using model 1')
print('   +TEMP_model2: Predicted temperature using model 2')
print('   +TEMP_model3: Predicted temperature using model 3')
print("You can access the result file on s3 with this link https://{0}.s3.amazonaws.com/{1} or you can open the file {1} in your current directory".format(s3_bucket, output_name))
print(df_res)
