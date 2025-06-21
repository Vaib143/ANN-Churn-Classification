import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

#Load the trained model,scaler pickle,onehot
model=load_model('model.h5') 

#Load the encoders and scalers
with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
#example input data
input_data={
    'CreditScore':600,
    'Geography':'France',
    'Gender':'Male',
    'Age': 40,
    'Tenure':3,
    'Balance':60000,
    'NumOfProducts':2,
    'HasCrCard':1,
    'IsActiveMember':1,
    'EstimatedSalary':500000
}


#***************One hot encoded Geography
print('\n')
geo_encoded=label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))
print(geo_encoded_df)


input_df=pd.DataFrame([input_data])
print(input_df)

#*********Encode categorial variables*****
input_df['Gender']=label_encoder_gender.transform(input_df['Gender'])
print(input_df)
print('\n')

#Concatination one hot encoded
input_dff=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)
print(input_dff)
print('\n')

#******scaling the input data**********

input_scaled=scaler.transform(input_dff)
print(input_scaled)
# prediction = model.predict(input_scaled)
# print("Prediction Probability:", prediction[0][0])
# print("Exited?" , "Yes" if prediction[0][0] > 0.5 else "No")



#*******Predict Cgurn*******
prediction = model.predict(input_scaled)
print(prediction)

prediction_prob=prediction[0][0]
print(prediction_prob)
if prediction_prob>0.5:
    print('Likely to churn')
else:
    print('Not likely to churn')