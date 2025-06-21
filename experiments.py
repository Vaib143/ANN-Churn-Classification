import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle

#********Load The Dataset***********

data = pd.read_csv(r"C:\Users\vaibh\Desktop\ANNClassification\Churn_Modelling.csv")
#print(data.head())

# ****************Preprocess the data,Drop irrelavant columns**************

data=data.drop(['RowNumber' , 'CustomerId','Surname'] , axis=1)
#print(data)
print('\n')

#************Encode Categoricalvariables********
# Gender turns int o the 0 and 1 form.

label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
#print(data)

#***********One-Hot-Encode Geography********

from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo=OneHotEncoder()
geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']])
#print(geo_encoder)
print('\n')

#print(geo_encoder.toarray())
print('\n')

#print(onehot_encoder_geo.get_feature_names_out(["Geography"]))   #  #['Geography_France' 'Geography_Germany' 'Geography_Spain']
      
print('\n')

geo_encoded_df=pd.DataFrame(geo_encoder.toarray(),columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))
#print(geo_encoded_df)
print('\n')
#*******Combine oneHot ENCODER COLUMNs with the original  data
data=pd.concat([data.drop('Geography',axis=1) , geo_encoded_df],axis=1)
#print(data.head())

#*****Save the Encoder And Scalers*******

# File Modes for pickle:
# 'wb': write binary (when saving)
# 'rb': read binary (when loading)

with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)
    

print('\n')  
    
#****** Divide the dataset into dependent and independent features******

X=data.drop('Exited',axis=1)
y=data['Exited']

#************Split the data in training and testing sets*******
#  random_state
# random_state is a seed value used by the random number generator.
# It ensures reproducibility.
# That means: you will get the same result every time you run the code, as long as the data and parameters stay the same.
# If you don’t set random_state, the results may change each time you run the code.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#***********Scale these features**********
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

#print(X_train)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)
    

#    ************ANN IMPLEMENTATION**********
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime


# **********Build our ANN Model********
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),   #HL1 connected with input layer
    Dense(32,activation='relu'),                                   #HL2 
    Dense(1,activation='sigmoid')            #output layer
]
)
print(model.summary())

import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()
print(loss)
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

#******Setup the Tensorboard******
from tensorflow.keras.callbacks import EarlyStopping ,TensorBoard
log_dir="logs/fit/" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])

#**********Setup EarlyStopping*****

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)


#*******Training the Model*********
history=model.fit (X_train,y_train,validation_data=(X_test,y_test),epochs=100,
                  callbacks=[tensorflow_callback, early_stopping_callback]
         )
# You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is 
# considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` 
# or `keras.saving.save_model(model, 'my_model.keras')`.   so we aslo usw model.save('model.keras)
model.save('model.h5')

#Load TensorBoard Extension
# %load_ext tensorboard

tensorboard --logdir=logs/fit                #run this command in new terminal we get out dashboard
