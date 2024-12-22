import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv, global_mean_pool


class GNN(torch.nn.Module):
    # Custom GNN class that defines PLAIG's GNN with specified parameters.
    def __init__(self, hidden_channels, num_layers, dropout_rate, num_node_features, num_edge_features,
                 num_ligand_features, num_pocket_features):
        super(GNN, self).__init__()
        # Set a random seed for reproducibility to ensure consistent results across runs.
        torch.manual_seed(12345)
        # Initialize the first graph convolutional layer that processes node features and edge features
        self.conv1 = GeneralConv(num_node_features, hidden_channels, in_edge_channels=num_edge_features)
        # Create additional convolutional layers that maintain the hidden channels as input/output dimensions.
        self.convs = torch.nn.ModuleList(
            [GeneralConv(hidden_channels, hidden_channels, in_edge_channels=num_edge_features) for _ in
             range(num_layers - 1)])
        # Linear layer to process global ligand features.
        self.global_fc_ligand = Linear(num_ligand_features, hidden_channels)
        # Linear layer to process global pocket features.
        self.global_fc_pocket = Linear(num_pocket_features, hidden_channels)
        # Fusion layer to combine processed graph, ligand, and pocket features.
        self.fusion_fc = Linear(3 * hidden_channels, hidden_channels)
        # Final linear layer to produce the final binding affinity prediction.
        self.lin = Linear(hidden_channels, 1)
        # Store the dropout rate for regularization during training.
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_attr, batch, ligand_features, pocket_features, return_embeddings):
        # Forward pass through the first convolutional layer, applying ReLU activation after.
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        # Pass through all subsequent convolutional layers, applying ReLU after each layer.
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr).relu()
        # Global mean pooling to aggregate node embeddings into global average embeddings for each graph in the batch.
        x = global_mean_pool(x, batch)
        # Process global ligand features through a fully connected layer.
        global_ligand_features_transformed = self.global_fc_ligand(ligand_features)
        # Process global pocket features through a fully connected layer.
        global_pocket_features_transformed = self.global_fc_pocket(pocket_features)
        # Concatenate the global graph embeddings, ligand features, and pocket features into a single tensor.
        x = torch.cat([x, global_ligand_features_transformed, global_pocket_features_transformed], dim=1)
        # Pass the concatenated features through another fully connected layer.
        x = self.fusion_fc(x)
        # Apply dropout for regularization, only during training.
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # If return_embeddings is False (during training), pass the output through the final linear layer to produce a
        # singular prediction
        if not return_embeddings:
            x = self.lin(x)
        return x  # Return the final embeddings from PLAIG's GNN (if return_embeddings is True)
