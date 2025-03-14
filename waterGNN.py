import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.data import Data

def parse_snapshot(snapshot_filename):
    with open(snapshot_filename, "r") as f:
        snapshot = json.load(f)

    nodes = snapshot["nodes"]
    edges = snapshot["edges"]

    node_names = sorted(nodes.keys())

    node_feature_keys = [
        "demand", "elevation", "head", "leak_status", "leak_area",
        "leak_discharge_coeff", "leak_demand", "pressure", "diameter",
        "level", "max_level", "min_level", "overflow"
    ]

    node_features = []

    for node_name in node_names:
        node = nodes[node_name]
        feature_vector = [node[feature] for feature in node_feature_keys]
        feature_vector.append(node["setting"])
        feature_vector += node["node_type"]
        node_features.append(feature_vector)

    node_features_data = torch.tensor(np.array(node_features, dtype=np.float32))

    node_to_index = {name: i for i, name in enumerate(node_names)}

    edge_index = torch.tensor([[node_to_index[edge["start"]], node_to_index[edge["end"]]] for edge in edges.values()], dtype=torch.long).t().contiguous()

    edge_feature_keys = [
        "base_speed", "flow", "headloss", "velocity", "roughness", "status", "setting", "diameter"
    ]

    edge_features = []
    for edge in edges.values():
        feature_vector = [edge[feature] for feature in edge_feature_keys]
        feature_vector += edge["link_type"]
        edge_features.append(feature_vector)

    edge_attr = torch.tensor(np.array(edge_features, dtype=np.float32))

    return Data(x=node_features_data, edge_index=edge_index, edge_attr=edge_attr)

class WaterGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(WaterGNN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def from_state_file(state_file, in_channels, hidden_channels, out_channels):
        model = WaterGNN(in_channels, hidden_channels, out_channels)
        model.load_state_dict(torch.load(state_file))
        return model

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_weight=None)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=None)
        return x
    

if __name__ == "__main__":
    data = parse_snapshot("snapshot_2.json")
    # Instantiate the GNN model
    
    in_channels = 20 #data.x.shape[1]
    print(data.x.shape[1])
    hidden_channels = 32
    out_channels = 1  
    
    model = WaterGNN(in_channels, hidden_channels, out_channels)

    
    output = model(data.x, data.edge_index, data.edge_attr)

    print("Model Output Shape:", output.shape)  
    
    #save the model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved")

    #load the model
    model = WaterGNN.from_state_file('model.pth', in_channels, hidden_channels, out_channels)