from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler
import numpy as np

# Setting models' seed
np.random.seed(120)

# Loading Footballers stats from .csv file
stats = pd.read_csv('PlayersStats.csv', encoding='Windows-1250', index_col=1, sep=';')

# Checking DataFrame info
print(stats.info())
print(stats.head())

# Data Preprocessing
rezerwowi = stats["Min"] < 500
stats = stats.loc[~rezerwowi]
stats = stats.drop(["Squad", "Comp", "Born", "Rk", "Nation", "Age", "MP", "Starts", "Min", "90s"], axis=1)
stats = stats.dropna()
stats["Pos"] = stats["Pos"].str.slice(stop=2)
print(stats.head())

# Data Spliting
X = stats.drop(['Pos'], axis=1).values
y = stats['Pos'].values
y = pd.Categorical(y).codes
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#Data Scaling
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)

# Shaping Neural Network
model = Sequential()
model.add(Dense(64, input_shape = (131,), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))

# Compiling Neural Network
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Implementing EarlyStopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4)

# Model Training
history=model.fit(X_test, y_test, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping], batch_size=128)