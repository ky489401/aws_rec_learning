import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Settings for dummy data
n_users = 1000
n_items = 500
n_interactions = 10000

# Generate random user-item interactions
np.random.seed(42)  # For reproducibility
user_ids = np.random.randint(0, n_users, size=n_interactions)
item_ids = np.random.randint(0, n_items, size=n_interactions)
timestamps = np.random.randint(1609459200, 1640995200, size=n_interactions)  # Random timestamps from 2021
ratings = np.random.randint(0, 2, size=n_interactions)  # Binary ratings: 0 or 1

# Create the DataFrame
dummy_data = pd.DataFrame({
    "user_id": user_ids,
    "item_id": item_ids,
    "timestamp": timestamps,
    "rating": ratings
})

class InteractionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data.iloc[idx]['user_id']
        item = self.data.iloc[idx]['item_id']
        rating = self.data.iloc[idx]['rating']
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(rating, dtype=torch.float)

# Create dataset and dataloader
train_dataset = InteractionDataset(dummy_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, users, items):
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=1)
        return self.mlp(x).squeeze()

# Settings
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate and move model to device
model = NCF(n_users, n_items).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(epochs):
    model.train()
    for users, items, ratings in train_loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(users, items)
            loss = nn.BCEWithLogitsLoss()(preds, ratings)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# 🔥 After training, save the model as TorchScript
model.eval()
example_users = torch.randint(0, n_users, (1,), dtype=torch.long).to(device)
example_items = torch.randint(0, n_items, (1,), dtype=torch.long).to(device)
scripted_model = torch.jit.trace(model, (example_users, example_items))
scripted_model.save("ncf.pt")   # <<< This is the model you will pass to torch-model-archiver
print("✅ Model successfully scripted and saved as ncf.pt")
