#ncf_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCFModel(nn.Module):
    """
    Improved Neural Collaborative Filtering (NCF) model for recommendation.
    Combines matrix factorization with multilayer perceptron.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[256, 128, 64]):
        """
        Initialize the NCF model.
        
        Args:
            num_users: Number of unique users in the dataset
            num_items: Number of unique items in the dataset
            embedding_dim: Size of the embedding vectors
            layers: List of layer dimensions for the MLP component
        """
        super(NCFModel, self).__init__()
        
        # User and item embedding layers for GMF
        self.user_gmf_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_gmf_embedding = nn.Embedding(num_items, embedding_dim)
        
        # User and item embedding layers for MLP
        self.user_mlp_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with better initialization
        nn.init.xavier_uniform_(self.user_gmf_embedding.weight)
        nn.init.xavier_uniform_(self.item_gmf_embedding.weight)
        nn.init.xavier_uniform_(self.user_mlp_embedding.weight)
        nn.init.xavier_uniform_(self.item_mlp_embedding.weight)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim  # Concatenated user and item embeddings
        
        for i, output_size in enumerate(layers):
            self.fc_layers.append(nn.Linear(input_size, output_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.BatchNorm1d(output_size))
            # Reduce dropout to prevent overregularization
            self.fc_layers.append(nn.Dropout(p=0.1))
            input_size = output_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1] + embedding_dim, 1)
        
        # Initialize linear layers properly
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        """
        Forward pass of the NCF model.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted probability of interaction
        """
        # Get GMF user and item embeddings
        user_gmf_embedding = self.user_gmf_embedding(user_indices)
        item_gmf_embedding = self.item_gmf_embedding(item_indices)
        
        # GMF part - use element-wise product
        gmf_vector = user_gmf_embedding * item_gmf_embedding
        
        # Get MLP user and item embeddings
        user_mlp_embedding = self.user_mlp_embedding(user_indices)
        item_mlp_embedding = self.item_mlp_embedding(item_indices)
        
        # Concatenate user and item embeddings for MLP
        mlp_vector = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=1)
        
        # Pass through MLP layers
        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)
        
        # Concatenate GMF and MLP parts
        prediction_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        
        # Output prediction
        logits = self.output_layer(prediction_vector)
        pred = self.sigmoid(logits)
        
        return pred

class RecommendationDataset(torch.utils.data.Dataset):
    """
    Dataset for training the NCF model.
    """
    def __init__(self, interactions_df):
        self.users = torch.tensor(interactions_df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(interactions_df['product_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(interactions_df['interaction'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'label': self.labels[idx]
        }