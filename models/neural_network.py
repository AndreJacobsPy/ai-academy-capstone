from keras import Sequential
from keras.api.layers import Dense, Dropout, Input
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import torch
import pandas as pd


# loading data
df = pd.read_csv('../data/feature_engineering.csv')
X = df.drop(columns='target')
y = df['target']


# splitting data
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

x_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

print(x_train.shape, x_test.shape)


# build model using keras
model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))
model.add(Dense(32, activation='relu', kernel_regularizer="l2"))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer="l2"))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer="l2"))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# add callback to stop training if validation recall does not improve
callback = EarlyStopping(monitor='val_recall', patience=20)

# compile model
adam = Adam(learning_rate=0.001)
model.compile(adam, "binary_crossentropy", metrics=["accuracy", "recall"])
summary = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=1000,
    callbacks=[callback],
)

print(f"Accuracy: {summary.history['accuracy'][-1]:.2f}")
print(f"Validation Accuracy: {summary.history['val_accuracy'][-1]:.2f}")
print(f"Recall: {summary.history['recall'][-1]:.2f}")
print(f"Validation Recall: {summary.history['val_recall'][-1]:.2f}")
