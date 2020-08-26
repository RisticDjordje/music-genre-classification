import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from keras import models, layers, callbacks
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data.csv')
data.head()
data = shuffle(data)

data = data.drop(['filename'], axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

call = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)

history = model.fit(
                    X_train,
                    y_train,
                    epochs=400,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[call],
                    shuffle=True
                    )

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy: ', test_acc)
