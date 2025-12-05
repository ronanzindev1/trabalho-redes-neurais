# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# %%
train_df = pd.read_csv("./datasets/mitbih_train.csv", header=None)
test_df = pd.read_csv("./datasets/mitbih_test.csv", header=None)
train_df.head()
# %%
train_df.shape

# %%
# Treino
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Teste
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values
# %%

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled

# %%

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

np.unique(clusters)

# %%
cluster_vs_label = pd.crosstab(clusters, y_train,
                               rownames=["Cluster (KMeans)"],
                               colnames=["Classe real"])
print(cluster_vs_label)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, s=4)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("K-Means (k=5) no espaço PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#%%
num_classes = len(np.unique(y_train))

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
y_train_cat

#%%
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#%%
history = model.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=20,
    batch_size=256,
    verbose=1
)
#%%
loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"Acurácia no conjunto de teste: {acc:.4f}")

#%%
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#%%
target_names = [
    "0 - Normal (N)",
    "1 - Supraventricular (S)",
    "2 - Ventricular (V)",
    "3 - Fusão (F)",
    "4 - Desconhecido (Q)"
]

print(classification_report(y_test, y_pred, target_names=target_names))