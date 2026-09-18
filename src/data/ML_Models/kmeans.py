import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"C:\Users\malli\Downloads\PlacementPredictionSystem-1\PlacementPredictionSystem_malli_updated\Placementpredictionsystem\data\placement_data (1).csv")

features={
    "CGPA",
    "Attendance",
    "Projects",
    "CodingTestScore"
}

x=df[features].dropna()

print("Selected Features:")
print(X.head())

scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)

kmeans=KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters=kmeans.fit_predict(X_scaled)

X["Cluster"]=clusters

print("\nCluster Assignment")

print(X.head(10))

centres_scaled=kmeans.cluster_centers_

centers_df=pd.DataFrame(
    centers,
    coloumns=features
)

print("\nCluster Centers:")
print(centres_df)

print("/nStudents in Eavh Cluster:")
print(x["Cluster"].value_counts().sort_index())

plt.figure(figsize=(8,6))

plt.scatter(
    X["CGPA"],
    X["CodingTestScore"],
    x=X["cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("K-Means Clustering of Students")

plt.show()