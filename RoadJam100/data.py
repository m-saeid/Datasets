import os
import numpy as np
import open3d as o3d
import torch
import json
from torch.utils.data import Dataset
#import plotly.graph_objects as go
import json
from sklearn.cluster import KMeans
import random
from tqdm import tqdm

from sklearn.metrics import silhouette_score

CLASS_MAP = {"car":0, "bus":1, "motorcycle":2}



def process_road_data(road_name, vehicle_data, num_points, pcd_path="/content/drive/MyDrive/PointCloud/vehicles300"):
    """
    Processes point cloud data for a given road, performs clustering, and identifies objects.

    Args:
        road_name (str): The name of the road (e.g., 'r0').
        vehicle_data (dict): Dictionary containing vehicle data from the JSON file.
        path (str, optional): Path to the point cloud data directory. Defaults to "/content/drive/MyDrive/PointCloud/vehicles300".
        num_points (int, optional): Desired number of points for each object. Defaults to 1024.

    Returns:
        dict: Output dictionary containing object labels and points in the specified format.
    """

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 1. Remove data with z-value of zero
    filtered_indices = np.where(points[:, 2] != 0)[0]
    filtered_points = points[filtered_indices]

    # 2. Project to xy plane
    xy_points = filtered_points[:, :2]

    # 3. KMeans clustering in 2D space
    kmeans = KMeans(n_clusters=6, random_state=0, n_init='auto', ).fit(xy_points)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # 4. Add z-values back
    clustered_points = np.concatenate([xy_points, filtered_points[:, 2][:, np.newaxis]], axis=1)

    # 5. Identify clusters and assign labels
    road_data = vehicle_data[road_name]
    vehicle_centers = [obj["box_center"] for obj in road_data]
    vehicle_labels = [obj["label"] for obj in road_data]

    cluster_labels = {}
    for i, cluster_center in enumerate(cluster_centers):
        distances = [np.linalg.norm(cluster_center - np.array(center[:2])) for center in vehicle_centers]
        closest_vehicle_index = np.argmin(distances)
        cluster_labels[i] = vehicle_labels[closest_vehicle_index]

    # 6. Create output dictionary and adjust point counts
    Output = {}
    for i in range(6):
        object_name = f"Object{i+1}"
        points = clustered_points[labels == i].tolist()
        
        # Adjust point count to num_points
        if len(points) > num_points:
            points = random.sample(points, num_points)  # Randomly remove excess points
        elif len(points) < num_points:
            # Repeat a random point until desired length is reached
            while len(points) < num_points:
                points.append(random.choice(points))
                
        Output[object_name] = {
            "Label": CLASS_MAP[cluster_labels[i]],
            "Points": points
        }

    return Output

def load_data(num_points, dataset_mode): # dataset:[RoadJam100_th0-2, RoadJam100_th1, RoadJam100_th3]
    all_data = []
    all_label = []
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    with open(os.path.join(DATA_DIR, dataset_mode, "all_vehicles_labels_centers.json"), 'r') as f:
        vehicle_data = json.load(f)    # { 'r0': { [ {'label':'car' , 'box_center':[0,0,0] ] }  ,  'r1': {}  ,  'r2': {}}

    for i in tqdm(range(100)):
        out = process_road_data(f'r{i}', vehicle_data=vehicle_data, num_points=num_points,
                                pcd_path=os.path.join(DATA_DIR, dataset_mode, f"r{i}.pcd"))
        
        for obj in out:
            all_data.append(out[obj]['Points'])
            all_label.append([out[obj]['Label']])
    all_data = np.array(all_data).astype('float32')
    all_label = np.array(all_label).astype('int64')
    return all_data, all_label


def normalize(data):
    min_coords = np.min(data, axis=0)
    max_coords = np.max(data, axis=0)

    # Calculate the scaling factor based on the largest dimension
    scale_factor = 2 / np.max(max_coords - min_coords)

    # Apply the scaling and shift to normalize
    normalized_points = (data - min_coords) * scale_factor - 1
    normalized_points = normalized_points - normalized_points.mean(axis=0)  # Center at (0, 0, 0)
    return normalized_points


class RoadJam100(Dataset):
    def __init__(self, num_points=1024, dataset_mode='RoadJam100_th0-2'):  # dataset:[RoadJam100_th0-2, RoadJam100_th1, RoadJam100_th3]
        self.data, self.label = load_data(num_points, dataset_mode)
        self.num_points = num_points      

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        pointcloud = normalize(pointcloud)
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



if __name__ == '__main__':
    train = RoadJam100(1024)
    for data, label in train:
        print('*', data.shape)
        print('*', label.shape)
        break