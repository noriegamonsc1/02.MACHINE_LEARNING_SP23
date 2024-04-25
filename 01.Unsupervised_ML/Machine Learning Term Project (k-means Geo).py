import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import arcpy
from arcgis.features import GeoAccessor, GeoSeriesAccessor
from arcgis import GIS

df = pd.read_excel(r'C:\Users\susta\Documents\MSU\ML_Data\PowerPlantsintheU_Export_TableToExcel.xlsx')
df.head() #.iloc[:,10:30]

X = df[['Latitude', 'Longitude']]
X

def calculate_inertia(data, k_range):
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return inertia

k_range = range(1, 11)  # Adjust range as needed
inertia = calculate_inertia(X, k_range)

plt.plot(k_range, inertia, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

optimal_k = 3 # (determine this from the elbow method plot)
kmeans = KMeans(n_clusters=optimal_k)
df['cluster'] = kmeans.fit_predict(X)

plt.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='viridis', label=f'Cluster')
plt.title('Power Plant Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'w', 'y', 'k']  # Add more colors if you have more than 8 clusters

for i in range(optimal_k):
    # Filter data for each cluster
    clustered_data = df[df['cluster'] == i]
    
    # Assuming 'x' and 'y' are the names of your columns
    plt.scatter(clustered_data['Longitude'], clustered_data['Latitude'], color=colors[i], label=f'Cluster {i}')

plt.title('K-Means Clustering')
plt.xlabel('Longitude')  # Replace with your actual label
plt.ylabel('Latitude')  # Replace with your actual label
plt.legend()
plt.show()

from arcgis.features import GeoAccessor, GeoSeriesAccessor
s_df = pd.DataFrame.spatial.from_xy(df, 'Longitude', 'Latitude')

output_path = r'C:\Users\susta\Documents\ArcGIS\Projects\CESAC_PhD_Project\MSU_ML_Class.gdb' # Path to your geodatabase
output_fc = 'geo_clustered_power_plants' # Name for the new feature class

# Export to feature class
s_df.spatial.to_featureclass(location=f"{output_path}\\{output_fc}")
