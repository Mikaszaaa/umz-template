from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os
import pandas as pd
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import numpy 

def create_baseline():

    model = Sequential()
    model.add(Dense(3, input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

scaler = MinMaxScaler()
r = pd.read_csv(os.path.join("train", "train.tsv"), header=None, names=[
                "price", "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_train = pd.DataFrame(r, columns=["rooms", "floor", "sqrMetres"])
Y_train = pd.DataFrame(r, columns=["price"])

estimator = KerasRegressor(build_fn=create_baseline, epochs=100, verbose=True)
X_t = scaler.fit_transform(X_train)
estimator.fit(X_t, Y_train)
predictions_train = estimator.predict(X_t)


p = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=[
                "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')

X_dev = pd.DataFrame(p, columns=["rooms", "floor", "sqrMetres"])
X_d = scaler.transform(X_dev)

Y_dev = pd.read_csv(os.path.join("dev-0", "expected.tsv"), header=None, names=["price"], sep='\t')

predictions_dev = estimator.predict(X_d)



with open(os.path.join("dev-0", "out.tsv"), 'w') as file:
    for prediction in predictions_dev:
        file.write(str(prediction[0]) + '\n')


l = pd.read_csv(os.path.join("test-A", "in.tsv"), header=None, names=[
                "isNew", "rooms", "floor", "location", "sqrMetres"], sep='\t')
X_test = pd.DataFrame(l, columns=["rooms", "floor", "sqrMetres"])
X_s = scaler.transform(X_test)
predictions_test = estimator.predict(X_s)


with open(os.path.join("test-A", "out.tsv"), 'w') as file:
    for prediction in predictions_test:
        file.write(str(prediction[0]) + '\n')
