import os
import json
import boto3
import pickle
import sklearn
import warnings
warnings.simplefilter("ignore")

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
bucket = "sagemaker-ap-south-1-573002217864"
key = "cust-churn-model/transformation/transformation.sav"
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    
    payload = process_data(event)
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode())
    predicted_label = 'True' if result > 0.39 else 'False'
    
    return predicted_label

def process_data(event):
    trans = pickle.loads(s3.Object(bucket, key).get()['Body'].read())
    event.pop('Phone')
    event['Area Code'] = int(event['Area Code'])
    obj_data = [[value for key,value in event.items() if key in trans['obj_cols']]]
    num_data = [[value for key,value in event.items() if key in trans['num_cols']]]
    
    obj_data = trans['One_Hot'].transform(obj_data).toarray()
    num_data = trans['scaler'].transform(num_data)
    
    obj_data = [str(i) for i in obj_data[0]]
    num_data = [str(i) for i in num_data[0]]
    
    data = obj_data + num_data
    return ",".join(data)
