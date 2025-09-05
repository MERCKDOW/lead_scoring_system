from google.cloud import storage
import os

# Setup
GCS_BUCKET = 'cdow'
GCS_FOLDER = 'leads_env'
LOCAL_BASE = f"/content/{GCS_FOLDER}"
HF_CACHE = f"{LOCAL_BASE}/hf_cache"
WHEELS_PATH = f"{LOCAL_BASE}/wheels"
REQ_PATH = f"{LOCAL_BASE}/requirements.txt"

os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(WHEELS_PATH, exist_ok=True)

client = storage.Client()
bucket = client.bucket(GCS_BUCKET)

# Download requirements.txt
blob = bucket.blob(f'{GCS_FOLDER}/requirements.txt')
blob.download_to_filename(REQ_PATH)

print(' Found and downloaded requirements.txt')

# Download wheels
for blob in bucket.list_blobs(prefix=f'{GCS_FOLDER}/wheels'):
    relative_path = blob.name.replace(f'{GCS_FOLDER}/', '')  # Strip GCS_FOLDER prefix
    local_path = os.path.join(LOCAL_BASE, relative_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
print('Wheels downloaded')

# Download Hugging Face model files
for blob in bucket.list_blobs(prefix=f'{GCS_FOLDER}/hf_cache'):
    relative_path = blob.name.replace(f'{GCS_FOLDER}/', '')
    local_path = os.path.join(LOCAL_BASE, relative_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
print(' Hugging Face model files downloaded')


import sys
import subprocess

def install_requirements(requirements_file, wheels_dir):#!pip install --no-index --find-links={WHEELS_PATH} -r {REQ_PATH}
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-linkss={WHEELS_PATH}', '-r',  {REQ_PATH}])

from google.cloud import storage
from google.cloud import bigquery

import os
import re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import gcsfs
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from itertools import islice
import gc
from datasets import load_dataset
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

# Initialize BigQuery client
# Set up GCS client
client = storage.Client()



import pyarrow.dataset as ds
import gcsfs


client = bigquery.Client()
project_id = "expanded-nebula-754"
dataset_id = "sandbox_crdow"
table_id = "leads_training_set_text_category"
table_ref = f"{project_id}.{dataset_id}.{table_id}"

# Use a wildcard * to shard the output into multiple files
destination_uri = "gs://cdow/leads_data/quality_leads_set-*.parquet"

extract_job = client.extract_table(
    table_ref,
    destination_uri,
    location="US",  # Adjust if your dataset is in a different location
    job_config=bigquery.job.ExtractJobConfig(
        destination_format="PARQUET"
    )
)

# Wait for job to complete
extract_job.result()
print(f"Exported {table_ref} to {destination_uri} in Parquet format")


# Initialize GCS filesystem
fs = gcsfs.GCSFileSystem()

# List all files in the bucket folder
all_files = fs.ls("cdow/leads_data")

# Filter only the Parquet shards for quality_leads_set
parquet_files = [
    f"gs://{file}" for file in all_files
    if file.startswith("cdow/leads_data/quality_leads_set-") and file.endswith(".parquet")
]

# Load the filtered Parquet files into a dataset using the GCS filesystem
dataset = ds.dataset(parquet_files, format="parquet", filesystem=fs)

# Convert to a PyArrow Table and then to a Pandas DataFrame
table = dataset.to_table()
df = table.to_pandas()





def preprocess_url(url):
    if pd.isna(url) or url.strip() == "":
        return "__MISSING__"

    # Remove protocol
    url = re.sub(r'^https?:\/\/', ' ', url)

    # Replace separators with space
    url = re.sub(r'[\/\.\?\=\&\%\:\;\_\-]', ' ', url)

    # Remove common file extensions and tokens m Body content 14382 id 9a63 4b7a 88f4 activityid 245b 42f0 bfff medium email
    url = re.sub(r'\b(html|php|aspx|www|RedirectTo|jsp|json|xml|utm|medium|source|sfmc|Banner|Imagecontent|listid|subscriberid|JobSubscriberBatchID|Body|Full_string|Logo Image URL|txt|@Logo Image URL)\b', ' ', url, flags=re.IGNORECASE)

    # Remove long alphanumeric strings (≥8 chars) that contain digits
    url = re.sub(r'\b(?=\w{8,})(?=\w*\d)\w+\b', ' ', url)
    # Remove exactly 2-digit numbers
    url = re.sub(r'\b\d{2}\b', ' ', url)

    # Remove exactly 3-character alphanumeric strings with at least one digit and one letter
    url = re.sub(r'\b(?=[a-zA-Z0-9]{3}$)(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{4}\b', ' ', url)
    url = re.sub(r'\b(?=[a-zA-Z0-9]{3}$)(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{3}\b', ' ', url)
    url = re.sub(r'\b(?=[a-zA-Z0-9]{3}$)(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]{2}\b', ' ', url)

    # Collapse multiple spaces
    url = re.sub(r'\s+', ' ', url).strip()

    return url

def process_url_list(url_list):
    # Handle NaN or None
    if url_list is None or isinstance(url_list, float) and pd.isna(url_list):
        return ["__MISSING__"]

    # Convert numpy array to list if needed
    if isinstance(url_list, np.ndarray):
        url_list = url_list.tolist()

    # Handle empty list
    if not url_list:
        return ["__MISSING__"]

    # Process each string
    return [preprocess_url(url) if isinstance(url, str) and url.strip() != "" else "__MISSING__" for url in url_list]




df['processed_url'] = df['a_url'].apply(process_url_list)


df['opportunity_id'] = df['opportunity_id'].astype(str)
columns = list(df.columns[1:10]) + [df.columns[12]]

df_features = df[columns]

print(df['opportunity_id'].is_unique)
# Count NaNs in each column
nan_counts = df.isna().sum()

# Count empty lists in columns that contain lists
empty_list_counts = {}
for col in columns:#df.columns:
    if df[col].apply(lambda x: isinstance(x, object)).any():
        empty_list_counts[col] = df[col].apply(lambda x: isinstance(x, object) and len(x) == 0).sum()


string_map = {}
id_counter = 0
len_es=25
for row in df['a_emailsubject']:
    for s in row:
        key = s[:len_es]
        if key not in string_map:
            string_map[key] = {
                'id': id_counter,
                'full_string': s,
                'key': key
            }
            id_counter += 1

# Step 2: Create a lookup dictionary for fast access
key_to_id = {k: v['id'] for k, v in string_map.items()}

# Step 3: Map each row to list of IDs
def map_to_ids(lst):
    if lst is None:
        return []
    return [key_to_id[s[:len_es]] for s in lst]

df['a_emailsubject_ids'] = df['a_emailsubject'].apply(map_to_ids)

# Optional: Create final dictionary
final_dict = {v['id']: {'full_string': v['full_string'], 'key': v['key']} for v in string_map.values()}
non_empty_valid_lists = df[
    df['processed_url'].apply(lambda x: isinstance(x, list) and len(x) > 0 and "__MISSING__" not in x)
]

string_map_url = {}
id_counter_url = 0
len_url = 80

for row in df['processed_url']:
    for s in row:
        key = s[:len_url]
        if key not in string_map_url:
            string_map_url[key] = {
                'id': id_counter_url,
                'full_string': s,
                'key': key
            }
            id_counter_url += 1

# Step 2: Create final_dict_url
final_dict_url = {
    v['id']: {'full_string': v['full_string'], 'key': v['key']}
    for v in string_map_url.values()
}

# Step 3: Create reverse lookup from key to ID using final_dict_url
key_to_id_url = {v['key']: k for k, v in final_dict_url.items()}

# Step 4: Map each row to list of IDs using the substring key
def map_to_ids_url(lst):
    if lst is None:
        return []
    return [key_to_id_url.get(s[:len_url], -1) for s in lst]  # -1 if not found

# Step 5: Apply to DataFrame
df['url_ids'] = df['processed_url'].apply(map_to_ids_url)




string_map_pi = {}
id_counter_pi = 0
len_pi=55
for row in df['a_product_interest']:
    for s in row:
        key = s[:len_pi]
        if key not in string_map_pi:
            string_map_pi[key] = {
                'id': id_counter_pi,
                'full_string': s,
                'key': key
            }
            id_counter_pi += 1

# Step 2: Create a lookup dictionary for fast access
key_to_id_ls = {k: v['id'] for k, v in string_map_pi.items()}

# Step 3: Map each row to list of IDs
def map_to_ids_pi(lst):
    if lst is None:
        return []
    return [key_to_id_ls[s[:len_pi]] for s in lst]

df['a_product_interest_ids'] = df['a_product_interest'].apply(map_to_ids_pi)

# Optional: Create final dictionary
final_dict_lr = {v['id']: {'full_string': v['full_string'], 'key': v['key']} for v in string_map_pi.values()}


string_map_lr = {}
id_counter_lr = 0
len_lr=55
for row in df['a_leads_recent']:
    for s in row:
        key = s[:len_lr]
        if key not in string_map_lr:
            string_map_lr[key] = {
                'id': id_counter_lr,
                'full_string': s,
                'key': key
            }
            id_counter_lr += 1

# Step 2: Create a lookup dictionary for fast access
key_to_id_ls = {k: v['id'] for k, v in string_map_lr.items()}

# Step 3: Map each row to list of IDs
def map_to_ids_lr(lst):
    if lst is None:
        return []
    return [key_to_id_ls[s[:len_lr]] for s in lst]

df['a_leads_ids'] = df['a_leads_recent'].apply(map_to_ids_lr)

# Optional: Create final dictionary
final_dict_lr = {v['id']: {'full_string': v['full_string'], 'key': v['key']} for v in string_map_lr.values()}





df_categories = df[[df.columns[0],df.columns[7],df.columns[11],df.columns[14],df.columns[15],df.columns[16],df.columns[17]]]


del df
import gc
gc.collect()
def extract_single_or_missing(lst):
    if not lst:
        return '__MISSING__'
    elif len(lst) == 1:
        return str(lst[0])
    else:
        return '_MULTI_'
df_categories['a_product_interest_cat'] = df_categories['a_product_interest_ids'].apply(extract_single_or_missing)
df_categories['a_leads_cat'] = df_categories['a_leads_ids'].apply(extract_single_or_missing)


df_categories['page_views'] =df_categories['page_views'].astype(float)


assert df_categories['a_is_won'].apply(lambda x:  isinstance(x, object) and len(x)).all()
invalid_rows = df_categories[~df_categories['a_is_won'].apply(lambda x: isinstance(x, object) and len(x) == 1)]

def force_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif hasattr(x, 'tolist'):
        return x.tolist()
    return x
df_categories['a_is_won'] = df_categories['a_is_won'].apply(force_list)
df_categories['a_is_won_float'] = df_categories['a_is_won'].apply(lambda x: float(x[0]))



df_XY = df_categories[['url_ids','a_emailsubject_ids','page_views','a_product_interest_cat','a_leads_cat','a_is_won_float']]
categorical_cols = ['a_leads_cat','a_product_interest_cat']

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(df_categories[categorical_cols])
#print(encoder.categories_)
categories = [len(cats) for cats in encoder.categories_]
for cats  in encoder.categories_:
    print(len(cats))

label_sum= df_XY['a_is_won_float'].sum()



PAD = 0

def shift_tokens(x):
    return [xi + 1 for xi in x] if len(x) > 0 else [PAD]
#print(all_tokens[0:100])
df_XY['a_emailsubject_ids'] = df_XY['a_emailsubject_ids'].apply(shift_tokens)



from collections import Counter

##CAN ONLY BE USED TO TEST MODELS PROPER MAPS SHOULD BE CREATED
##
##
all_ids = [id for sublist in df_XY['a_emailsubject_ids'] for id in sublist]
frequency = Counter(all_ids)

# Step 2: Define a function to replace infrequent IDs
def replace_infrequent_ids(id_list, freq_counter, threshold=900, replacement=1231):
    return [id if freq_counter[id] >= threshold else replacement for id in id_list]

# Step 3: Apply the function to the column
df_XY['a_emailsubject_ids'] = df_XY['a_emailsubject_ids'].apply(lambda x: replace_infrequent_ids(x, frequency))
all_ids = [id for sublist in df_XY['a_emailsubject_ids'] for id in sublist]


# Step 1: Get all unique IDs from the column
unique_ids = set(id for sublist in df_XY['a_emailsubject_ids'] for id in sublist)

# Step 2: Create a mapping from original ID to new compact ID
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_ids))}

# Step 3: Apply the mapping to each list in the column
df_XY['a_emailsubject_ids'] = df_XY['a_emailsubject_ids'].apply(
    lambda id_list: [id_mapping[id] for id in id_list]
)

all_ids = [id for sublist in df_XY['a_emailsubject_ids'] for id in sublist]
frequency = Counter(all_ids)


frequency_df = pd.DataFrame(frequency.items(), columns=['ID', 'Frequency']).sort_values(by='Frequency', ascending=False)
'''
### WORKING MODEL ####
class CustomTabTransformer(nn.Module):
    def __init__(self, categories, num_continuous, token_vocab_size=25, token_emb_dim=64, gru_hidden_dim=128):
        super().__init__()

        # TabTransformer base
        self.tab = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=64,
            dim_out=128,
            depth=4,
            heads=4
        )

        # Token embedding + GRU + attention
        self.token_emb = nn.Embedding(token_vocab_size, token_emb_dim, padding_idx=0)
        self.gru = nn.GRU(token_emb_dim, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(gru_hidden_dim * 2, 1)  # for attention weights
        self.final = nn.Linear(128 + gru_hidden_dim * 2, 1)  # final output
        combined_dim = 128 + gru_hidden_dim * 2
#        # GRU path
        #print(self.tab.embedding_dim)

        # Projection layers to same dimension
        self.tab_proj = nn.Linear(128, combined_dim)#self.tab.embedding_dim
        self.seq_proj = nn.Linear(gru_hidden_dim * 2, combined_dim)

        # Gating
        self.gate = nn.Linear(combined_dim, combined_dim)

        # Final output
        #self.head = nn.Linear(combined_dim, output_dim)

        

    def forward(self, x_categ, x_cont, token_seqs, token_lengths):
        #tab_out = self.tab(x_categ, x_cont)  # shape: [B, 128]

        tab_emb = self.tab(x_categ, x_cont)  # shape: [B, 128]
        tab_emb = self.tab_proj(tab_emb)
        # Embed and pack token sequences
        token_embs = self.token_emb(token_seqs)  # [B, T, D]
        packed = nn.utils.rnn.pack_padded_sequence(token_embs, token_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T, 2H]

        # Attention over GRU outputs
        attn_weights = torch.softmax(self.attn(unpacked).squeeze(-1), dim=1)  # [B, T]
        attn_out = torch.sum(unpacked * attn_weights.unsqueeze(-1), dim=1)  # [B, 2H]
        attn_out = self.seq_proj(attn_out)

        # Gating
        g = torch.sigmoid(self.gate(tab_emb)) # [batch, combined_dim]
        combined = g * tab_emb + (1 - g) * attn_out
        # Combine TabTransformer + GRU-attn output
        #combined = torch.cat([tab_out, attn_out], dim=1)  # [B, 128 + 2H]
        return self.final(combined).squeeze(-1)  # [B]


from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TabTokenDataset(Dataset):
    def __init__(self, df, token_pad_len=15):
        self.df = df
        self.token_pad_len = token_pad_len
        self.product = pd.Categorical(df['a_product_interest_cat']).codes
        self.leads = pd.Categorical(df['a_leads_cat']).codes
        self.subjects = df['a_emailsubject_ids'].apply(
            lambda x: list(map(int, x)) + [0] * (token_pad_len - len(x))
        ).tolist()
        self.page_views = df['page_views'].values.astype('float32')
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x_cont  = torch.tensor(self.page_views[idx] , dtype=torch.float32)#.values.astype('float32')'a_emailsubject_ids')
        x_categ = torch.tensor([ self.leads[idx],self.product[idx]], dtype=torch.long)
        tokens = torch.tensor(self.subjects[idx][:self.token_pad_len], dtype=torch.long)
        token_len = torch.tensor(len(tokens), dtype=torch.long)
        padded_tokens = F.pad(tokens, (0, self.token_pad_len - len(tokens)), value=0)
        target = torch.tensor(row["a_is_won_float"], dtype=torch.float32)
        return x_categ, x_cont, padded_tokens, token_len, target

dataset = TabTokenDataset(df_XY)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


'''

### WORKING MODEL ####
class CustomTabTransformer(nn.Module):
    def __init__(self, categories, num_continuous, token_vocab_size=25, token_emb_dim=64, gru_hidden_dim=128):
        super().__init__()

        # TabTransformer base
        self.tab = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=64,
            dim_out=128,
            depth=4,
            heads=4
        )

        # Token embedding + GRU + attention
        self.token_emb = nn.Embedding(token_vocab_size, token_emb_dim, padding_idx=0)
        self.gru = nn.GRU(token_emb_dim, gru_hidden_dim, batch_first=True, bidirectional=True)

        self.attn = nn.Linear(gru_hidden_dim * 2, 1)  # for attention weights
        self.final = nn.Linear(128 + gru_hidden_dim * 2, 1)  # final output

    def forward(self, x_categ, x_cont, token_seqs, token_lengths):
        tab_out = self.tab(x_categ, x_cont)  # shape: [B, 128]

        # Embed and pack token sequences
        token_embs = self.token_emb(token_seqs)  # [B, T, D]
        packed = nn.utils.rnn.pack_padded_sequence(token_embs, token_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T, 2H]

        # Attention over GRU outputs
        attn_weights = torch.softmax(self.attn(unpacked).squeeze(-1), dim=1)  # [B, T]
        attn_out = torch.sum(unpacked * attn_weights.unsqueeze(-1), dim=1)  # [B, 2H]

        # Combine TabTransformer + GRU-attn output
        combined = torch.cat([tab_out, attn_out], dim=1)  # [B, 128 + 2H]
        return self.final(combined).squeeze(-1)  # [B]

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TabTokenDataset(Dataset):
    def __init__(self, df, token_pad_len=15):
        self.df = df
        self.token_pad_len = token_pad_len
        self.product = pd.Categorical(df['a_product_interest_cat']).codes
        self.leads = pd.Categorical(df['a_leads_cat']).codes
        self.subjects = df['a_emailsubject_ids'].apply(
            lambda x: list(map(int, x)) + [0] * (token_pad_len - len(x))
        ).tolist()
        self.page_views = df['page_views'].values.astype('float32')
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x_cont  = torch.tensor(self.page_views[idx] , dtype=torch.float32)#.values.astype('float32')'a_emailsubject_ids')
        x_categ = torch.tensor([ self.leads[idx],self.product[idx]], dtype=torch.long)
        tokens = torch.tensor(self.subjects[idx][:self.token_pad_len], dtype=torch.long)
        token_len = torch.tensor(len(tokens), dtype=torch.long)
        padded_tokens = F.pad(tokens, (0, self.token_pad_len - len(tokens)), value=0)
        target = torch.tensor(row["a_is_won_float"], dtype=torch.float32)
        return x_categ, x_cont, padded_tokens, token_len, target

dataset = TabTokenDataset(df_XY)
loader = DataLoader(dataset, batch_size=512, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_tokens = df_XY['a_emailsubject_ids'].explode()
email_vocab = all_tokens.max() + 2
print(email_vocab)
del all_tokens
gc.collect()

model = CustomTabTransformer(categories=categories, num_continuous=1,token_vocab_size=email_vocab).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
torch.autograd.set_detect_anomaly(True)
epochs = 30
#



for epoch in range(epochs):
  for x_categ, x_cont, tokens, token_lens, targets in loader:
      x_categ = x_categ.to(device)
      x_cont = x_cont.to(device).unsqueeze(1)
      tokens = tokens.to(device)
      token_lens = token_lens.to(device)
      targets = targets.to(device)
      #token_lens, targets = x_categ.to(device), x_cont.to(device).unsqueeze(), tokens.to(device), token_lens.to(device), targets.to(device)

      optimizer.zero_grad()
      logits = model(x_categ, x_cont, tokens, token_lens)
      loss = loss_fn(logits, targets)
      loss.backward()
      optimizer.step()

  print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")



  batch_size = 1024

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model.eval()
err = 0

#with open("/content/results.txt", "w") as f:
#    for i in range(n_samples):
#        l, p = data[i]
#        choice = p > 0.5
#        f.write(f"label = {l}, prediction = {p}, choice = {choice}\n")

with open("/content/results.txt", "w") as f:
  with torch.no_grad():
    for x_categ, x_cont, tokens, token_lens, targets in loader:
        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device).unsqueeze(1)
        tokens = tokens.to(device)
        token_lens = token_lens.to(device)
        targets = targets.to(device)
        pred = model(x_categ, x_cont, tokens, token_lens)
        for p,l in zip(torch.sigmoid(pred),targets):
          #print(f"predicted {float(p.squeeze().cpu().item() > 0.5):.4f}   label {l.squeeze().cpu().item():.4f}")
          pp = float(p.squeeze().cpu().item())
          ll =  float(l.squeeze().cpu().item())
          choice = pp> 0.5
          f.write(f"label = {ll}, prediction = {pp}, choice = {choice}\n")
          err = err + abs(pp - ll)
        break
    print(err/batch_size)