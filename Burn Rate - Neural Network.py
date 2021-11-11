import pandas as pd
import numpy as np

df = pd.read_csv('Burn Rate.csv')

# Cleaning Data!
df.dropna(inplace = True)

df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
df['Company Type'] = df['Company Type'].apply(lambda x: 0 if x == 'Service' else 1)
df['WFH Setup Available'] = df['WFH Setup Available'].apply(lambda x: 0 if x == 'No' else 1)
df['Resource Allocation'] = df['Resource Allocation'].astype(int)

# Normalize
df['Designation Normalize'] = (df['Designation'] - df['Designation'].min()) / (df['Designation'].max() - df['Designation'].min()) 
df['Resource Allocation Normalize'] = (df['Resource Allocation'] - df['Resource Allocation'].min()) / (df['Resource Allocation'].max() - df['Resource Allocation'].min())
df['Mental Fatigue Score Normalize'] = (df['Mental Fatigue Score'] - df['Mental Fatigue Score'].min()) / (df['Mental Fatigue Score'].max() - df['Mental Fatigue Score'].min())

# Split Data
X = df[['Gender', 
        'Company Type', 
        'WFH Setup Available',
        'Designation Normalize', 
        'Resource Allocation Normalize',
        'Mental Fatigue Score Normalize']]
y = df['Burn Rate']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Artificial Neural Network
X_train_sample = X_train[:2000]
X_test_sample = X_test[:500]
y_train_sample = y_train[:2000]
y_test_sample = y_test[:500]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

X_train_sample = np.array(X_train_sample)
X_test_sample = np.array(X_test_sample)
y_train_sample = np.array(y_train_sample)
y_test_sample = np.array(y_test_sample)

model = Sequential()

model.add(Dense(X_train_sample.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer=Adam(0.00001), loss='mse')

r = model.fit(X_train_sample, y_train_sample,
              validation_data=(X_test_sample,y_test_sample),
              batch_size=1,
              epochs=100)

test_pred = model.predict(X_test_sample)
train_pred = model.predict(X_train_sample)

from ann_visualizer.visualize import ann_viz
ann_viz(model, view=True, filename="network.gv", title="MyNeural Network");