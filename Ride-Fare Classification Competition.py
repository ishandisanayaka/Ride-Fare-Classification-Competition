import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D,BatchNormalization,Convolution1D,Conv1D,MaxPooling1D
from keras.optimizers import SGD,RMSprop
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


import pandas as pd

from sklearn.metrics import accuracy_score



df=pd.read_csv(r'C:\Users\DELL\Downloads\train(3).csv')
df_predict1=pd.read_csv(r'C:\Users\DELL\Downloads\test(2).csv')
df_predict1.pickup_time=pd.to_datetime(df_predict1.pickup_time)
df_predict1.drop_time=pd.to_datetime(df_predict1.drop_time)

df_predict1["pickup_time"]=df_predict1["pickup_time"].dt.hour+df_predict1["pickup_time"].dt.minute*0.01
df_predict1["pick_dayOfWeek"]=df_predict1["drop_time"].dt.day_name()

mapping1 = {'Monday': 1,'Tuesday': 2,'Wednesday': 3,'Thursday': 4,'Friday': 5,'Saturday': 6,'Sunday': 7,'':8}


df_predict1.pick_dayOfWeek = [mapping1[item] for item in df_predict1.pick_dayOfWeek]

df_predict1["drop_time"]=df_predict1["drop_time"].dt.hour+df_predict1["drop_time"].dt.minute*0.01

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

df_predict1["meter_waitinga_and_free"]=df_predict1["meter_waiting_fare"]*(df_predict1["meter_waiting_till_pickup"]+df_predict1["meter_waiting"])
df_predict1["meter_waitinga_and_free3"]=df_predict1["meter_waiting_fare"]*df_predict1["meter_waiting_till_pickup"]
df_predict1["meter_waitinga_and_free2"]=df_predict1["meter_waiting_fare"]*df_predict1["meter_waiting"]



df_predict1["distance"]=((df_predict1["drop_lat"]-df_predict1["pick_lat"])**2 + (df_predict1["drop_lon"]-df_predict1["pick_lon"])**2)**0.5
df_predict1["distance2"]=((df_predict1["drop_lat"]-df_predict1["pick_lat"]) + (df_predict1["drop_lon"]-df_predict1["pick_lon"]))


df_predict=df_predict1.drop('pickup_time',axis=1).drop('drop_time',axis=1).drop('pick_lat',axis=1).drop('pick_lon',axis=1).drop('drop_lat',axis=1).drop('drop_lon',axis=1)#.drop('meter_waiting',axis=1).drop('meter_waiting_till_pickup',axis=1)

dff=df
"""
dff=df.fillna({
    'additional_fare':10.5,
    'meter_waiting_fare':0
    })
"""
dff=dff.fillna(df.groupby('label').mean())
dff.pickup_time=pd.to_datetime(dff.pickup_time)
dff.drop_time=pd.to_datetime(dff.drop_time)

dff["pickup_time"]=dff["pickup_time"].dt.hour+dff["pickup_time"].dt.minute*0.01
dff["pick_dayOfWeek"]=dff["drop_time"].dt.day_name()

mapping = {'Monday': 1,'Tuesday': 2,'Wednesday': 3,'Thursday': 4,'Friday': 5,'Saturday': 6,'Sunday': 7,'':8}

dff.pick_dayOfWeek = [mapping[item] for item in dff.pick_dayOfWeek]


dff["drop_time"]=dff["drop_time"].dt.hour+dff["drop_time"].dt.minute*0.01


dff["meter_waitinga_and_free"]=dff["meter_waiting_fare"]*(dff["meter_waiting_till_pickup"]+dff["meter_waiting"])
dff["meter_waitinga_and_free3"]=dff["meter_waiting_fare"]*dff["meter_waiting_till_pickup"]
dff["meter_waitinga_and_free2"]=dff["meter_waiting_fare"]*dff["meter_waiting"]
dff["correct_fair"]=dff["fare"]+dff["additional_fare"]+dff["meter_waiting_fare"]

from math import sqrt
dff["distance"]=((dff["drop_lat"]-dff["pick_lat"])**2 + (dff["drop_lon"]-dff["pick_lon"])**2)**0.5
dff["distance2"]=((dff["drop_lat"]-dff["pick_lat"]) + (dff["drop_lon"]-dff["pick_lon"]))

dff=dff.fillna(df.groupby('label').mean())
dff=dff.fillna(dff.mean())


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

x=dff.drop('label',axis=1).drop('pickup_time',axis=1).drop('drop_time',axis=1).drop('pick_lat',axis=1).drop('pick_lon',axis=1).drop('drop_lat',axis=1).drop('drop_lon',axis=1)#.drop('meter_waiting',axis=1).drop('meter_waiting_till_pickup',axis=1)


y=dff['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y,train_size=0.9,shuffle=True,stratify=y,random_state=6)

from xgboost import XGBClassifier


model_XGB = XGBClassifier( n_estimators=1500)
model_XGB.fit(x_train,y_train)
predic71=model_XGB.predict(x_test)
print(accuracy_score(y_test, predic71))




scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x = scaler.transform(x)
df_predict = scaler.transform(df_predict)


model2 = Sequential()
model2.add(Dense(1028, input_dim=10, activation='relu'))
model2.add(Dropout(0.5))
#model2.add(Dense(128, activation='relu'))
#model2.add(Dropout(0.5))
model2.add(Dense(564, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(232, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))
sgd=SGD(lr=0.01,momentum=0.1)
# compile the keras model
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model2.fit(x, y, epochs=500, batch_size=32, verbose=1, validation_data=(x_test, y_test))
# evaluate the keras model
_, accuracy = model2.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

prediction1 = model2.predict_classes(x_test)
print(f1_score(y_test, prediction1, average='macro'))


"""
model1=Sequential()
model1.add(Dense(128,activation='relu',input_dim=11))
model1.add(Dropout(0.1))
model1.add(Dense(64,activation='relu'))
model1.add(Dropout(0.1))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1,activation='softmax'))


sgd=SGD(lr=0.01,momentum=0.9,nesterov=True)
model1.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
print(x_train.shape)
model1.fit(x, y,epochs=1, batch_size=228)
predictionn=model1.predict(x_test)
#print(accuracy_score(y_test, predictionn))
_, accuracy = model1.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
"""
"""
print(x_train.shape)

model_mm=Sequential()
model_mm.add(Convolution1D(32,3,border_mode='same',input_shape=[12]))
model_mm.add(Activation('relu'))
model_mm.add(Convolution2D(32,3,3))
model_mm.add(Activation('relu'))
model_mm.add(MaxPooling2D(pool_size=(2,2)))
model_mm.add(Dropout(0.25))

model_mm.add(Convolution2D(32,3,3,border_mode='same'))
model_mm.add(Activation('relu'))
model_mm.add(Convolution2D(64,3,3))
model_mm.add(Activation('relu'))
model_mm.add(MaxPooling2D(pool_size=(2,2)))
model_mm.add(Dropout(0.25))

model_mm.add(Flatten())
model_mm.add(512)
model_mm.add(Activation('relu'))
model_mm.add(Dropout(0.5))
model_mm.add(Dense(1))
model_mm.add(Activation('softmax'))

model_mm.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

model_mm.fit(x_train,y_train,batch_size=128,epoch=15,validation_data=(x_test,y_test),suffle=True)

print(model_mm.evaluate(x_test,y_test))


verbose, epochs, batch_size = 0, 10, 32
n_timesteps, n_features, n_outputs = x_train.shape[0], x_train.shape[1], y_train.shape[0]
modela = Sequential()
modela.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
modela.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
modela.add(Dropout(0.5))
modela.add(MaxPooling1D(pool_size=2))
modela.add(Flatten())
modela.add(Dense(100, activation='relu'))
modela.add(Dense(n_outputs, activation='softmax'))
modela.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
modela.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = modela.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print(accuracy)

"""
#s = pd.Series(dff["pickup_time"], index=range(len(dff["pickup_time"])))
"""
#s.plot()
import tensorflow as tf
from tensorflow import keras
modelz = keras.Sequential([
    keras.layers.Flatten(input_shape=(12,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

modelz.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

modelz.fit(x, y, epochs=1)
test_loss, test_acc = modelz.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
"""


"""

import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network

model3 = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='adam', 
                                                 alpha=0.01, batch_size='auto', learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, 
                                                 max_iter=10000, shuffle=True, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)

    # Train the model on the whole data set
model3.fit(x, y)

predictions_s = model3.predict(x_train)
print(predictions_s)
accuracy = sklearn.metrics.accuracy_score(y_train, predictions_s)
print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
print('Classification Report:')
print(sklearn.metrics.classification_report(y_train, predictions_s))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(y_train, predictions_s))
print('')

    # Evaluate on test data
print('\n---- Test data ----')
predictions = model3.predict(x_test)
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
print('Classification Report:')
print(sklearn.metrics.classification_report(y_test, predictions))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(y_test, predictions))
"""

##########################
"""
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier,VotingClassifier


from sklearn.ensemble import GradientBoostingClassifier
model_G = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1500, max_depth=5,  subsample=0.8)

# fit the model with the training data
model_G.fit(x_train,y_train)
predic61=model_G.predict(x_test)
print(accuracy_score(y_test, predic61))



from lightgbm import LGBMClassifier

model_LGMC = LGBMClassifier(n_estimators=1500,max_depth=6)
#model_LGMC = XGBClassifier()
model_LGMC.fit(x_train,y_train)
predic72=model_LGMC.predict(x_test)
print(accuracy_score(y_test, predic72))

from xgboost import XGBClassifier
model_XGB = XGBClassifier( learning_rate=0.1, n_estimators=1500, max_depth=5,min_child_weight=2, gamma=0.2, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
model_XGB.fit(x_train,y_train)
predic71=model_XGB.predict(x_test)
print(accuracy_score(y_test, predic71))

from catboost import CatBoostClassifier
model_catboost = CatBoostClassifier(verbose=1500, n_estimators=2000)
model_catboost.fit(x_train,y_train)
predic73=model_catboost.predict(x_test)
print(accuracy_score(y_test, predic73))
"""
####################
"""
evc=VotingClassifier(estimators=[('model_LGMC',model_LGMC),('model_G',model_G)])
evc.fit(x_train,y_train)
predictio=evc.predict(x_test)
print(accuracy_score(y_test, predictio))
"""
"""
bg=BaggingClassifier(GradientBoostingClassifier(n_estimators=5,max_depth=5),max_samples=0.5,max_features=1.0,n_estimators=100)
bg.fit(x_train,y_train)
predict=bg.predict(x_test)
print(accuracy_score(y_test, predict))

adb=AdaBoostClassifier(GradientBoostingClassifier(n_estimators=5,max_depth=5),n_estimators=10,learning_rate=1)
adb.fit(x_train,y_train)
predicti=adb.predict(x_test)
print(accuracy_score(y_test, predicti))
"""
from sklearn.tree import DecisionTreeClassifier
"""
bg=BaggingClassifier(Sequential(),max_samples=0.5,max_features=1.0,n_estimators=100)
bg.fit(x_train,y_train)
predict=bg.predict(x_test)
print(accuracy_score(y_test, predict))


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train, y_train)
prediction=model.predict(x_test)
print(accuracy_score(y_test, prediction))
#sdboost
"""
"""##########################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier,VotingClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=101,max_leaf_nodes=6)
rf.fit(x_train,y_train)
predic=rf.predict(x_test)
print(accuracy_score(y_test, predic))


rf1=DecisionTreeClassifier(criterion="entropy", max_depth=10,min_samples_split=0.3, min_samples_leaf=2, max_features=9,)
rf1.fit(x_train,y_train)
prediccc=rf1.predict(x_test)
print(accuracy_score(y_test, prediccc))

bg=BaggingClassifier( DecisionTreeClassifier(),max_samples=5,max_features=1.0,n_estimators=100)
#bg=BaggingClassifier(max_samples=0.5,max_features=1.0,n_estimators=1000)
bg.fit(x_train,y_train)
predict=bg.predict(x_test)
print(accuracy_score(y_test, predict))
"""
"""
adb=AdaBoostClassifier(CatBoostClassifier(verbose=1500, n_estimators=10),n_estimators=100)
adb.fit(x_train,y_train)
predicti=adb.predict(x_test)
print(accuracy_score(y_test, predicti))
"""
"""
lr=LogisticRegression()
dt=DecisionTreeClassifier()
evc=VotingClassifier(estimators=[('bg',bg),('dt',dt)],voting='hard')
evc.fit(x_train,y_train)
predictio=evc.predict(x_test)
print(accuracy_score(y_test, predictio))

adb=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)
adb.fit(x_train,y_train)
predicti=adb.predict(x_test)
print(accuracy_score(y_test, predicti))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
lr=LogisticRegression()
dt=DecisionTreeClassifier()
svm=SVC(kernel='poly',degree=2)
evc=VotingClassifier(estimators=[('lr',lr),('dt',dt),('svm',svm)],voting='hard')
evc.fit(x_train,y_train)
predictio=evc.predict(x_test)
print(accuracy_score(y_test, predictio))



from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)
nn.fit(x_train,y_train)
predic2=nn.predict(x_test)
print(accuracy_score(y_test, predic2))

#from sklearn.svm import SVC
#svc=SVC(kernel='linear')
#svc.fit(x_train,y_train)
#predic3=nn.predict(x_test)
#print(accuracy_score(y_test, predic3))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7,metric='euclidean',p=2)
knn.fit(x_train,y_train)
predic4=knn.predict(x_test)
print(accuracy_score(y_test, predic4))

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
BerNB=BernoulliNB(binarize=True)
BerNB.fit(x_train,y_train)
predic5=BerNB.predict(x_test)
print(accuracy_score(y_test, predic5))

MultiNB=MultinomialNB()
MultiNB.fit(x_train,y_train)
predic6=MultiNB.predict(x_test)
print(accuracy_score(y_test, predic6))

GauNB=GaussianNB()
GauNB.fit(x_train,y_train)
predic7=GauNB.predict(x_test)
print(accuracy_score(y_test, predic7))

from sklearn import tree
treecl=tree.DecisionTreeClassifier()
treecl.fit(x_train,y_train)
predic8=treecl.predict(x_test)
print(accuracy_score(y_test, predic8))

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
nn1 = MLPRegressor(hidden_layer_sizes=(20,10,10,10,5,),activation='relu',max_iter=200)
nn1.fit(x_train,y_train)
predic9=nn1.predict(x_test)
#print(accuracy_score(y_test, predic9))

from sklearn.linear_model import SGDClassifier
sdg=SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
sdg.fit(x_train,y_train)
predic11=sdg.predict(x_test)
print(accuracy_score(y_test, predic11))

from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(x_train,y_train)
#predic12=rbf_feature.predict(x_test)
#print(accuracy_score(y_test, predic12))

"""
"""
predicted_data=model2.predict_classes(df_predict)
predic_output=[]
for i in predicted_data:
    predic_output.append(i[0])
#print(predic_output)
 
predicted_data2=model1.predict_classes(df_predict)
predic_output2=[]
for i in predicted_data2:
    predic_output2.append(i[0])

data=pd.DataFrame({'tripid':df_predict1['tripid'],'prediction':predic_output})
data.to_csv('sample_submission_neural_network.csv',index=False)


data2=pd.DataFrame({'tripid':df_predict1['tripid'],'prediction':predic_output2})
data2.to_csv('sample_submission_neural_network_part_2.csv',index=False)
"""
"""
predicted_data3=modelz.predict_classes(df_predict)
#print(predicted_data3)
predic_output_sgsgfhgsj=[]
predic_output2=[]
for i in predicted_data3:
    predic_output2.append(i[0])
    #if(i[0]>=0.602):
    #    predic_output2.append(1)
    #if(i[0]<0.602):
     #   predic_output2.append(0)
    
#print(predic_output2)
data5=pd.DataFrame({'tripid':df_predict1['tripid'],'prediction':predic_output2})
data5.to_csv('sample_submission_neural_network_modelz.csv',index=False)


"""
"""
predicted_data4=model_catboost.predict(df_predict)
print(predicted_data4)
data6=pd.DataFrame({'tripid':df_predict1['tripid'],'prediction':predicted_data4})
data6.to_csv('sample_submission_neural_network_model_catboost.csv',index=False)

predicted_data5=evc.predict(df_predict)
print(predicted_data5)
data6=pd.DataFrame({'tripid':df_predict1['tripid'],'prediction':predicted_data5})
data6.to_csv('sample_submission_neural_network_model_evc.csv',index=False)
"""

predictions = model2.predict_classes(df_predict)
print(predictions)
predic_output2=[]
for i in predictions:
    predic_output2.append(i[0])
data6=pd.DataFrame({'tripid':df_predict1['tripid'],'prediction':predic_output2})
data6.to_csv('sample_submission_neural_network_nuralnet.csv',index=False)

