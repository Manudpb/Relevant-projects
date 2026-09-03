import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import random
import os


random.seed(50)
np.random.seed(50)
torch.manual_seed(50)


# === Load dataset ===
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
item_meta = pd.read_csv( 'item_meta.csv', engine='python', on_bad_lines='skip')

# === Remove duplicated interactions ===
def deduplicate_user_sequence(df):
    deduped_rows = []
    for user_id, group in df.groupby('user_id'):
        seen = set()
        sorted_group = group.sort_values('timestamp')
        for _, row in sorted_group.iterrows():
            if row['item_id'] not in seen:
                seen.add(row['item_id'])
                deduped_rows.append(row)
    return pd.DataFrame(deduped_rows)

df = deduplicate_user_sequence(df)
test_df = deduplicate_user_sequence(test_df)
train_df = df

# === Encode users/items ===
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

all_user_ids = pd.concat([train_df['user_id'], test_df['user_id']]).unique()
all_item_ids = pd.concat([train_df['item_id'], test_df['item_id'], item_meta['item_id']]).unique()

user_encoder.fit(all_user_ids)
item_encoder.fit(all_item_ids)

train_df['user_idx'] = user_encoder.transform(train_df['user_id'])
test_df['user_idx'] = user_encoder.transform(test_df['user_id'])

train_df['item_idx'] = item_encoder.transform(train_df['item_id'])
test_df['item_idx'] = item_encoder.transform(test_df['item_id'])
item_meta['item_idx'] = item_encoder.transform(item_meta['item_id'])

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

# === TF-IDF for titles ===
tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(item_meta['title'].fillna(''))
title_tfidf = tfidf_matrix.toarray()

# === Category encoding and rating scaling ===
cat_encoder = LabelEncoder()
item_meta['cat_idx'] = cat_encoder.fit_transform(item_meta['main_category'].fillna('Unknown'))
rating_scaled = MinMaxScaler().fit_transform(item_meta[['average_rating']].fillna(0))

title_features = torch.zeros((num_items, title_tfidf.shape[1]))
cat_features = torch.zeros((num_items,), dtype=torch.long)
rating_features = torch.zeros((num_items, 1))

for _, row in item_meta.iterrows():
    idx = row['item_idx']
    title_features[idx] = torch.tensor(title_tfidf[row.name])
    cat_features[idx] = row['cat_idx']
    rating_features[idx] = torch.tensor(rating_scaled[row.name], dtype=torch.float32)

# === Train Dataset with negative sampling(10 negatives per interaction) ===
class NCFDataset(Dataset):
    def __init__(self, df, num_items, num_negatives=10):
        self.user_item_set = set(zip(df['user_idx'], df['item_idx']))
        self.users = df['user_idx'].values
        self.items = df['item_idx'].values
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.data = []
        self._prepare()

    def _prepare(self):
        for u, i in zip(self.users, self.items):
            self.data.append((u, i, 1))
            for _ in range(self.num_negatives):
                j = random.randint(0, self.num_items - 1)
                while (u, j) in self.user_item_set:
                    j = random.randint(0, self.num_items - 1)
                self.data.append((u, j, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, label = self.data[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# === NCF model ===
class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_cat, title_dim, emb_dim=64, hidden_layers=[128, 64]):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.cat_emb = nn.Embedding(num_cat, 8)
        self.title_features = title_features
        self.cat_features = cat_features
        self.rating_features = rating_features

        input_dim = emb_dim * 2 + title_dim + 8 + 1
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, item_idx):
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)
        title_feat = self.title_features[item_idx]
        cat_feat = self.cat_emb(self.cat_features[item_idx])
        rating_feat = self.rating_features[item_idx]
        x = torch.cat([user_emb, item_emb, title_feat, cat_feat, rating_feat], dim=1)
        x = self.mlp(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x.squeeze()

# === Prepare dataloaders and model ===
train_dataset = NCFDataset(train_df, num_items, num_negatives=10)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

model = NCF(num_users, num_items, len(cat_encoder.classes_), title_features.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

user_train_items = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

# Generate 99 negative items per user for evaluation on test set
def sample_eval_items(df, num_items, num_neg=99):
    user_pos_neg = {}
    for user_idx, group in df.groupby('user_idx'):
        seen = user_train_items.get(user_idx, set())
        pos_items = group['item_idx'].tolist()
        user_pos_neg[user_idx] = []

        for pos_item in pos_items:
            negatives = set()
            while len(negatives) < num_neg:
                neg = random.randint(0, num_items - 1)
                if neg not in seen and neg != pos_item:
                    negatives.add(neg)
            user_pos_neg[user_idx].append((pos_item, list(negatives)))
    return user_pos_neg

test_eval_items = sample_eval_items(test_df, num_items, num_neg=99)

# === NDCG evaluation ===
def evaluate_ndcg_sampled(model, eval_items_dict, k=10):
    model.eval()
    ndcg_scores = []
    with torch.no_grad():
        for user_idx, pos_neg_list in tqdm(eval_items_dict.items()):
            for pos_item, negatives in pos_neg_list:
                items = [pos_item] + negatives
                labels = [1] + [0] * len(negatives)

                user_tensor = torch.tensor([user_idx] * len(items))
                item_tensor = torch.tensor(items)

                scores = model(user_tensor, item_tensor).cpu().numpy()
                labels = np.array(labels)

                sorted_idx = np.argsort(-scores)
                sorted_labels = labels[sorted_idx]
                sorted_scores = scores[sorted_idx]

                ndcg = ndcg_score([sorted_labels[:k]], [sorted_scores[:k]])
                ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

# === Training ===
for epoch in range(3):
    model.train()
    total_loss = 0
    for user_idx, item_idx, label in tqdm(train_loader):
        optimizer.zero_grad()
        pred = model(user_idx, item_idx)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}: Train loss = {avg_loss:.4f}")

# === Final test evaluation ===
test_ndcg = evaluate_ndcg_sampled(model, test_eval_items, k=10)
print(f"Test NDCG@10: {test_ndcg:.4f}")

sample_sub = pd.read_csv('sample_submission.csv')
sample_sub['user_idx'] = user_encoder.transform(sample_sub['user_id'])

def recommend_top_10(model, user_idx, num_items, user_train_items, k=10):
    model.eval()
    seen_items = user_train_items.get(user_idx, set())
    candidate_items = [i for i in range(num_items) if i not in seen_items] # Exclude seen items

    user_tensor = torch.tensor([user_idx] * len(candidate_items))
    item_tensor = torch.tensor(candidate_items)
    # Predict scores for all candidate items, excluding interacted items for that user
    with torch.no_grad():
        scores = model(user_tensor, item_tensor).cpu().numpy()
    
    top_k_idx = np.argsort(-scores)[:k]
    top_k_items = [candidate_items[i] for i in top_k_idx]
    return top_k_items

submission_rows = []

for user_id, user_idx in zip(sample_sub['user_id'], sample_sub['user_idx']):
    top_items_idx = recommend_top_10(model, user_idx, num_items, user_train_items, k=10)
    top_items = item_encoder.inverse_transform(top_items_idx)
    top10_item_ids = top_items.tolist()
    
    submission_rows.append({
        'ID': user_id,
        'user_id': user_id,
        'item_id': ','.join(map(str, top10_item_ids))
    })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)
