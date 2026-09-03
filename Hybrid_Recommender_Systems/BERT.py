import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import os
import random


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# === Load dataset ===
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
item_meta = pd.read_csv('item_meta.csv', engine='python', on_bad_lines='skip')

# === Remove duplicated interactions per user ===
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
train_df = df.copy()

# === Item encoder ===
all_item_ids = pd.concat([df['item_id'], test_df['item_id'], item_meta['item_id']]).unique()
item_encoder = LabelEncoder()
item_encoder.fit(all_item_ids)

# Special token for MASK
MASK_IDX = len(item_encoder.classes_)
num_items = MASK_IDX + 1


for d in [train_df, test_df, item_meta]:
    d['item_idx'] = item_encoder.transform(d['item_id'])


tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(item_meta['title'].fillna(''))
title_tfidf = tfidf_matrix.toarray()

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

MAX_SEQ_LEN = 10
MASK_PROB = 0.15

class MaskedRecDataset(Dataset):
    def __init__(self, df, mask_all=False):
        self.samples = []
        for user_id, group in df.groupby('user_id'):
            item_seq = group.sort_values('timestamp')['item_idx'].tolist()
            if len(item_seq) < 2:
                continue
            item_seq = item_seq[-MAX_SEQ_LEN:]
            seq = [0] * (MAX_SEQ_LEN - len(item_seq)) + item_seq
            input_seq = seq.copy()
            labels = [-100] * MAX_SEQ_LEN
            for i in range(MAX_SEQ_LEN):
                if seq[i] == 0:
                    continue
                if mask_all or random.random() < MASK_PROB:
                    labels[i] = seq[i]
                    input_seq[i] = MASK_IDX
            if any(l != -100 for l in labels):
                self.samples.append((input_seq, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# === Transformer-based Model ===
class BERT4RecModel(nn.Module):
    def __init__(self, num_items, num_cat, title_dim, emb_dim=64, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.cat_emb = nn.Embedding(num_cat, 8)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim + title_dim + 8 + 1, num_items)

    def forward(self, seq):
        positions = torch.arange(seq.size(1)).unsqueeze(0)
        item_embs = self.item_emb(seq) + self.pos_emb(positions)
        x = self.transformer(item_embs.transpose(0, 1)).transpose(0, 1)

        last_items = seq
        title_feat = title_features[last_items]
        cat_feat = self.cat_emb(cat_features[last_items])
        rating_feat = rating_features[last_items]

        x = torch.cat([x, title_feat, cat_feat, rating_feat], dim=2)
        logits = self.fc(x)
        return logits

# === Data loaders ===
train_dataset = MaskedRecDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


model = BERT4RecModel(num_items, len(cat_encoder.classes_), title_features.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# === Evaluation===
def evaluate_with_negatives(model, loader, num_negatives=100, print_examples=True, num_prints=50):
    model.eval()
    all_preds = []
    all_targets = []
    recall_hits = 0
    total_relevant = 0
    printed = 0

    all_item_set = set(range(len(item_encoder.classes_)))

    with torch.no_grad():
        for x, y in loader:
            batch_size, seq_len = y.shape
            logits = model(x)
            y_np = y.cpu().numpy()

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    true_item = y_np[batch_idx, seq_idx]
                    if true_item == -100:
                        continue

                    neg_candidates = list(all_item_set - {true_item})
                    sampled_negs = random.sample(neg_candidates, num_negatives)
                    candidates = [true_item] + sampled_negs
                    scores = logits[batch_idx, seq_idx, candidates].cpu().numpy()
                    ranked_indices = np.argsort(-scores)

                    rel = np.zeros(len(candidates))
                    rel[0] = 1
                    pred_rel = rel[ranked_indices]

                    all_preds.append(pred_rel)
                    all_targets.append(np.array([1] + [0]*num_negatives))

                    if 0 in ranked_indices[:10]:
                        recall_hits += 1
                    total_relevant += 1


    avg_ndcg = np.mean([ndcg_score([t], [p]) for t, p in zip(all_targets, all_preds)])
    recall_at_10 = recall_hits / total_relevant if total_relevant > 0 else 0.0

    return avg_ndcg, recall_at_10

# === Training loop ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, num_items), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

# === Test evaluation ===
test_dataset = MaskedRecDataset(test_df, mask_all=True)
test_loader = DataLoader(test_dataset, batch_size=128)
test_ndcg, test_recall10 = evaluate_with_negatives(model, test_loader, num_negatives=99)
print(f"Test NDCG@10: {test_ndcg:.4f} Recall@10: {test_recall10:.4f}")


sample_sub = pd.read_csv('sample_submission.csv')
users_to_predict = sample_sub['user_id'].unique()

model.eval()
submission_rows = []

with torch.no_grad():
    for user_id in users_to_predict:
        user_train_data = train_df[train_df['user_id'] == user_id].sort_values('timestamp')
        item_seq = user_train_data['item_idx'].tolist()

        # Add mask token at end of train seq for predicting next item
        full_seq = item_seq + [MASK_IDX]

        # Trunqate or pad to MAX_SEQ_LEN
        if len(full_seq) > MAX_SEQ_LEN:
            seq = full_seq[-MAX_SEQ_LEN:]
        else:
            seq = [0] * (MAX_SEQ_LEN - len(full_seq)) + full_seq

        seq_tensor = torch.tensor([seq], dtype=torch.long) 

        logits = model(seq_tensor)  

        # Get prediction at the masked position (last one)
        masked_pos = seq.index(MASK_IDX)
        logits = logits[0, masked_pos]

        # Exclude already seen items
        seen_items = set(item_seq)
        logits[list(seen_items)] = float('-inf')

        top10_idx = torch.topk(logits, 10).indices.cpu().numpy()
        top10_item_ids = item_encoder.inverse_transform(top10_idx)

        submission_rows.append({
            'ID': user_id,
            'user_id': user_id,
            'item_id': ','.join(map(str, top10_item_ids))
        })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)


