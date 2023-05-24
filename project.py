from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setting models' seed
np.random.seed(32)

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
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(128, input_shape = (131,), activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32)

# Adding RandomizedSearchCV to find best parameters
params = dict(optimizer=['sgd', 'adam'], batch_size=[16,32,64,128], activation=['relu','tanh', 'leaky_relu'])
random_search = RandomizedSearchCV(model, param_distributions=params, cv=3, n_iter=8)

# Implementing EarlyStopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Searching for the best parameters and showing results
random_search_results = random_search.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stopping])
print('Best: {} using: {}'.format(random_search_results.best_score_, random_search_results.best_params_))

