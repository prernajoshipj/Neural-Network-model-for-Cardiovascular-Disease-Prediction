import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import os


# Load dataset
df = pd.read_csv(r'app\cardio_train.csv', sep=";")

# Drop ID column
df.drop(columns=['id'], inplace=True)

# One-hot encode categorical features
glucose = pd.get_dummies(df['gluc'], prefix='gluc', drop_first=True)
chol = pd.get_dummies(df['cholesterol'], prefix='cholesterol', drop_first=True)

df = pd.concat([df, glucose, chol], axis=1)
df.drop(['gluc', 'cholesterol'], axis=1, inplace=True)

# Split dataset
X = df.drop(['cardio'], axis=1)
y = df['cardio'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA for Dimensionality Reduction
pca = PCA(.90)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Define Neural Network Model
classifier = Sequential()
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile Model
classifier.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
classifier.fit(X_train, y_train, batch_size=43, epochs=79)

# Create save directory
save_dir = "saved_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save Model in .h5 format
classifier.save(os.path.join(save_dir, "cardio_model.h5"))
print("Model saved successfully in 'saved_model/cardio_model.h5'!")
