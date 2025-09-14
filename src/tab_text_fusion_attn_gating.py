import os
import re
import sys
import gcsfs
import torch
import gc
import subprocess

from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler






from google.cloud import storage
from google.cloud import bigquery


from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple,Union


from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM

from itertools import islice


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

client = storage.Client()
bucket = client.bucket('cdow')



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
import pyarrow.dataset as ds
import gcsfs
fs = gcsfs.GCSFileSystem()

# List all files in the bucket folder
all_files = fs.ls("cdow/leads_data")
#print(all_files)

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
    url = re.sub(r'\b(html|php|aspx|www|RedirectTo|jsp|json|xml|ex2|ex|utm|medium|source|sfmc|Banner|Imagecontent|listid|subscriberid|JobSubscriberBatchID|Body|Full_string|Logo Image URL|txt|@Logo Image URL)\b', ' ', url, flags=re.IGNORECASE)

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


from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
import json
class QuantileCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, max_len=40, missing_token="__MISSING__",
                 empty_token="__EMPTY__", rare_quantile=0.1):
        """
        cols: list of column names to encode
        max_len: maximum substring length per token
        missing_token: category for rare/unseen strings
        empty_token: category for empty lists
        rare_quantile: bottom quantile of frequency to bin into missing_token
        """
        self.cols = cols
        self.max_len = max_len
        self.missing_token = missing_token
        self.empty_token = empty_token
        self.rare_quantile = rare_quantile
        self.vocabs = {}
        self.counter = []

    def _normalize(self, arr):
        """Normalize list/ndarray/scalar to a truncated string token."""
        if arr is None:
            return self.empty_token

        if isinstance(arr, (list, np.ndarray)):
            if len(arr) == 0:
                return self.empty_token
            s = " ".join(str(x) for x in arr)
        else: # scalar string, number, etc.
            s = str(arr)

        return s[: self.max_len]

    def fit(self, X, y=None):
        """Build vocabularies based on frequency quantiles."""
        X = pd.DataFrame(X) # ensure dataframe
        for col in self.cols:
            # Count normalized strings
            self.counter = Counter(self._normalize(val) for val in X[col])

            freqs = np.array(list(self.counter.values()))
            threshold = np.quantile(freqs, self.rare_quantile)

            vocab = {}
            for token, count in self.counter.items():
                if count > threshold and token not in (self.empty_token, self.missing_token):
                    vocab[token] = len(vocab) + 1 # reserve 0 for padding if needed

            # Add special tokens
            vocab[self.empty_token] = len(vocab) + 1
            vocab[self.missing_token] = len(vocab) + 1

            self.vocabs[col] = vocab

        return self

    def transform(self, X):
        """Transform columns into integer IDs."""
        X = pd.DataFrame(X).copy()
        for col in self.cols:
            vocab = self.vocabs[col]
            miss_id = vocab[self.missing_token]
            empty_id = vocab[self.empty_token]

            def encode(val):
                s = self._normalize(val)
                if s == self.empty_token:
                    return empty_id
                return vocab.get(s, miss_id)

            X[col] = X[col].apply(encode)

        return X

    def inverse_transform(self, X):
        """Convert integer IDs back to strings (best effort)."""
        X = pd.DataFrame(X).copy()
        for col in self.cols:
            inv_vocab = {v: k for k, v in self.vocabs[col].items()}
            X[col] = X[col].apply(lambda i: inv_vocab.get(i, self.missing_token))
        return X

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.vocabs, f)

    def load(self, path):
        with open(path, "r") as f:
            self.vocabs = json.load(f)




def force_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif hasattr(x, 'tolist'):
        return x.tolist()
    return x

df['a_is_won'] = df['a_is_won'].apply(force_list)
df['a_emailsubject'] = df['a_emailsubject'].apply(force_list)
df['days_open'] =df['days_open'].astype(float)
df['page_views '] = df['page_views'].astype(float)
df['a_is_won_float'] = df['a_is_won'].apply(lambda x: float(x[0]))


cont_cols = ['page_views', 'days_open']
text_cols = ['processed_url', 'a_emailsubject']
target    = ["a_is_won_float"]

cat_cols = ['a_search_terms','a_product_interest','a_campaign_name','a_Type','a_leads_recent','a_currency']
encoder = QuantileCategoricalEncoder(cols=cat_cols,max_len=20, rare_quantile=0.2)

df_encoded = encoder.fit_transform(df).copy()
all_cols = cat_cols + cont_cols + text_cols + target


df_set = df_encoded[all_cols].copy()
#def to_scalar_str(x):
#    if isinstance(x, np.ndarray):
#        x = x.tolist()
#    if isinstance(x, list):
#        # choose a stable representation; join as string
#        return "|".join(map(str, x))
#    return str(x)

#for c in cat_cols:
#    df_set[c] = df_set[c].apply(to_scalar_str)




os.environ["HF_HUB_OFFLINE"] = "1"
os.environ['TRANSFORMERS_CACHE'] = "./hf_cache_smol/"
model_name =  "HuggingFaceTB/SmolLM2-135M" 

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
enc.fit(df_set[cat_cols])                             # learn categories
df_set[cat_cols] = enc.transform(df_set[cat_cols]) 
categories = [len(cats) for cats in enc.categories_]
for cats  in enc.categories_:
    print(len(cats))
    
print(cat_cols)


class CustomTabTransformer(nn.Module):
    def __init__(self, categories=categories, num_continuous=1,output_dim=512):#token_vocab_size=25,, dropout=0.1 token_emb_dim=128, gru_hidden_dim=128,
        super().__init__()

        # TabTransformer base
        self.tab = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=64,
            dim_out=512,
            depth=6,
            heads=4
        )
        combined_dim = output_dim# + gru_hidden_dim * 2
        self.output_dim = output_dim
      
       # Projection layers to same dimension
        self.tab_proj = nn.Linear(512, 128)   #should BN or add dropout? 128=dim_out
        self.tab_relu = nn.ReLU()
        self.tab_norm = nn.LayerNorm(128)
    def forward(self,cats,conts):#, token_seqs, token_lengths,print_gate):
        
        tab_emb  = self.tab(cats, conts)  # shape: [B, 128]
        tab_emb  = self.tab_proj(tab_emb)
        tab_emb  = self.tab_relu(tab_emb)
        tab_emb = self.tab_norm(tab_emb)
        return tab_emb         
        
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


POOL_NUM_QUERIES = 1
POOL_OUT_DIM = 512

import torch
import torch.nn as nn
import math
class PrototypeLayer(nn.Module):
    """
    Differentiable clustering layer with learnable prototypes.
    - Training mode (.train()): soft assignment (convex combination of prototypes).
    - Eval mode (.eval()): hard nearest-prototype lookup.
    """
    def __init__(self, embed_dim, num_prototypes=128, temperature=0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        # Codebook: learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))

    def forward(self, x):
        """
        x: (batch, embed_dim)
        returns:
            clustered: (batch, embed_dim) clustered representation
            weights: (batch, num_prototypes) assignment distribution
        """
        # Similarity scores
        sims = torch.matmul(x, self.prototypes.T) # (batch, num_prototypes)

        if self.training:
            # Soft assignment (differentiable)
            weights = F.softmax(sims / self.temperature, dim=-1)
            clustered = torch.matmul(weights, self.prototypes)
        else:
            # Hard assignment (inference)
            indices = torch.argmax(sims, dim=-1) # (batch,)
            clustered = self.prototypes[indices] # (batch, embed_dim)
            weights = F.one_hot(indices, num_classes=self.num_prototypes).float()

        return clustered, weights


class MultiQueryPooling(nn.Module):
    def __init__(self, hidden_size: int, num_queries: int = 1, out_dim: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries

        self.queries  = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)  # [Q, H]
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.val_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Project concatenated [Q*H] to out_dim
        self.out_proj = nn.Linear(num_queries * hidden_size, out_dim)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [B, N, H]
        attention_mask: [B, N] (1=valid, 0=pad)
        returns: [B, out_dim]
        """
        B, N, H = hidden_states.shape
        K = self.key_proj(hidden_states)   # [B, N, H]
        V = self.val_proj(hidden_states)   # [B, N, H]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, H]

        # scores: [B, Q, N]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(H)

        # mask: [B, 1, N]
        mask = attention_mask.unsqueeze(1)  # broadcast across Q
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # if a row is all -inf, softmax -> NaN; avoid that by replacing -inf with big negative
        # or clamp with torch.where. If you know there's always at least one real token, we can skip.
        attn = torch.softmax(scores, dim=-1)  # [B, Q, N]

        pooled = torch.matmul(attn, V)        # [B, Q, H]
        pooled_flat = pooled.reshape(B, -1)   # [B, Q*H]
        out = self.out_proj(pooled_flat)      # [B, out_dim]
        return out
        
def tokenize_into_overlapping_chunks(
    text,
    tokenizer,
    max_len,                       # total length per chunk INCLUDING special tokens
    overlap=0,                     # overlap in CONTENT tokens
    max_chunks=None,               # cap #chunks (e.g., 2 or 3)
    max_total_content_tokens=None, # optional: global cap BEFORE chunking (content only)
):
    """
    Returns a list of dicts: [{"input_ids": [...], "attention_mask": [...]}, ...]
    - Sliding windows with overlap over CONTENT tokens
    - Special tokens are added per chunk
    - Each returned chunk length <= max_len (including special tokens)
    """
    # 1) Encode content WITHOUT special tokens
    content_ids = tokenizer.encode(text, add_special_tokens=False)

    # Optional global cap on content (if you really want it)
    if max_total_content_tokens is not None:
        content_ids = content_ids[:max_total_content_tokens]

    if len(content_ids) == 0:
        # Return a single empty chunk; the downstream pad() will handle it
        return [{"input_ids": [], "attention_mask": []}]

    # 2) How many special tokens get added per chunk?
    # (For BERT-like models this is typically 2: [CLS] and [SEP])
    special_tokens_per_chunk = tokenizer.num_special_tokens_to_add(pair=False)

    # 3) Compute the maximum number of CONTENT tokens we can fit per chunk
    max_content_per_chunk = max_len - special_tokens_per_chunk
    if max_content_per_chunk <= 0:
        raise ValueError(
            f"max_len={max_len} is too small for your tokenizer; "
            f"needs at least {special_tokens_per_chunk + 1}."
        )

    # 4) Determine stride from overlap
    # overlap is in CONTENT tokens. So stride = max_content_per_chunk - overlap
    stride = max(1, max_content_per_chunk - overlap)

    chunks = []
    start = 0
    created = 0

    while start < len(content_ids):
        if max_chunks is not None and created >= max_chunks:
            break

        end = min(start + max_content_per_chunk, len(content_ids))
        content_slice = content_ids[start:end]

        # Add special tokens per chunk
        chunk_ids = tokenizer.build_inputs_with_special_tokens(content_slice)
        chunk_mask = [1] * len(chunk_ids)

        chunks.append({"input_ids": chunk_ids, "attention_mask": chunk_mask})
        created += 1

        if end == len(content_ids):
            break
        start += stride

    if len(chunks) == 0:
        chunks.append({"input_ids": [], "attention_mask": []})

    return chunks

# --- Fixed MultiTextEncoder ---
class MultiTextEncoder(nn.Module):
    """
    Encodes multiple text fields with a shared transformer and per-field attention pooling.

    Returns for each field:
      - tensor of shape [batch_size, max_chunks, hidden_dim]
      - chunk_mask shape [batch_size, max_chunks] (1 for real chunk, 0 for padded chunk)
    """
    def __init__(self, model_name, text_fields):
        super().__init__()

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ['TRANSFORMERS_CACHE'] = "./hf_cache_smol/"
        model_name =  "./hf_cache_smol/"  
        self.field_names = text_fields
        self.proj_dim = 512
        # Parameters you had
        self.MAX_TOKENS_PER_SAMPLE = 64*8
        self.CHUNK_LEN = 16
        self.OVERLAP_TOKENS = 3
        self.MAX_CHUNKS = (self.MAX_TOKENS_PER_SAMPLE + self.CHUNK_LEN - 1) // self.CHUNK_LEN # -> 8

        self.MAX_CHUNKS = 4
        self.text_fields = text_fields

        # load tokenizer and model (local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = tokenizer
        tokenizer.model_max_length = self.MAX_TOKENS_PER_SAMPLE

        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        model.config.use_cache = False
        # ensure pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

                
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"

        # Only needed if you *added* tokens, not if you reuse eos as pad
        model.resize_token_embeddings(len(tokenizer))
        # Freeze all model parameters


        ### ALLOW TRIANING ---- UNCOMMENT TO FREEZE 
        #for param in model.parameters():
        #    param.requires_grad = False

        self.transformer = model
        #print(model.config.hidden_size)
        
        hidden_dim = self.transformer.config.hidden_size
        #print(hidden_dim)

        #self.poolers = nn.ModuleDict({
        #    field: MultiQueryPooling(hidden_dim,1,hidden_dim) for field in text_fields
        #})
        self.poolers = nn.ModuleDict({
            field: MultiQueryPooling(self.transformer.config.hidden_size, num_queries=1, out_dim=128)
            for field in text_fields
        })

        self.tab_proj = nn.Linear(128 , 64)   #should BN or add dropout? 128=dim_out
        self.tab_relu = nn.ReLU()
        self.tab_norm = nn.LayerNorm(64)
        
    def _encode_field_batch(self, texts, field_name):

        import torch
    
        device = next(self.transformer.parameters()).device
        B = len(texts)
        MAX_CHUNKS = self.MAX_CHUNKS
    
        # --- 1) Build overlapping chunks per sample ---
        all_chunk_items = []
        chunks_per_sample = []
        for text in texts:
            chunks = tokenize_into_overlapping_chunks(
                text,
                tokenizer=self.tokenizer,
                max_len=self.CHUNK_LEN,        # total per-chunk length incl specials
                overlap=self.OVERLAP_TOKENS,   # e.g., 100 (content overlap)
                max_chunks=self.MAX_CHUNKS,     # e.g., 2 or 3
                max_total_content_tokens = self.MAX_TOKENS_PER_SAMPLE 
            )
    
            # Ensure tokenizer.pad() can handle empty chunks. If a chunk has no tokens,
            # replace with a single pad token and mask=0 so the model ignores it.
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            for c in chunks:
                if "input_ids" not in c or c["input_ids"] is None:
                    c["input_ids"] = []
                if "attention_mask" not in c or c["attention_mask"] is None:
                    c["attention_mask"] = [1] * len(c["input_ids"])
    
                if len(c["input_ids"]) == 0:
                    c["input_ids"] = [pad_id]
                    c["attention_mask"] = [0]
    
                all_chunk_items.append(c)
    
            chunks_per_sample.append(len(chunks))
    
        total_chunks = sum(chunks_per_sample)
    
        # If absolutely no chunks exist (e.g., B==0), produce zeros with correct D
        pooler = self.poolers[field_name]
        # D is the final embedding size produced by your pooler (its out_proj)
        if hasattr(pooler, "out_proj"):
            D = pooler.out_proj.out_features
        else:
            # fallback if your pooler returns H (concatenated or not)
            D = self.transformer.config.hidden_size
        if total_chunks == 0:
            return (
                torch.zeros(B, MAX_CHUNKS, D, device=device),
                torch.zeros(B, MAX_CHUNKS, dtype=torch.long, device=device),
            )
    
        # --- 2) Pad and run the transformer ONCE over all chunks ---
        padded = self.tokenizer.pad(
            all_chunk_items,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        input_ids      = padded["input_ids"].to(device)         # [total_chunks, Lmax]
        attention_mask = padded["attention_mask"].to(device)    # [total_chunks, Lmax]
    
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        last_hidden_state = (
            outputs.hidden_states[-1]
            if outputs.hidden_states is not None
            else outputs.last_hidden_state
        )  # [total_chunks, Lmax, H]
    
        # --- 3) Pool each chunk to a fixed D-dim vector ---
        # pooler expects [B, N, H] and mask [B, N]
        chunk_vecs = pooler(last_hidden_state, attention_mask)  # [total_chunks, D]
        D = chunk_vecs.shape[-1]
    
        # --- 4) Regroup vectors back to [B, MAX_CHUNKS, D] + mask ---
        chunk_embs = torch.zeros(B, MAX_CHUNKS, D, device=device)
        chunk_mask = torch.zeros(B, MAX_CHUNKS, dtype=torch.long, device=device)
    
        offset = 0
        for b, n in enumerate(chunks_per_sample):
            n_use = min(n, MAX_CHUNKS)
            if n_use > 0:
                chunk_embs[b, :n_use] = chunk_vecs[offset : offset + n_use]
                chunk_mask[b, :n_use] = 1
            offset += n
    
        return chunk_embs, chunk_mask



    def forward(self, texts_dict):
        outputs = {}
        for field in self.text_fields:
            emb, mask = self._encode_field_batch(texts_dict[field], field)
            outputs[field] = (emb, mask)
    
        # --- NEW: reduce each [B, C, H] → [B, H] ---
        field_vecs = []
        for field, (emb, mask) in outputs.items():
            # simple mean across chunks
            #print(emb.size())
            emb = self.tab_proj(emb.mean(dim=1))
            emb = self.tab_relu(emb)
            emb = self.tab_norm(emb)
            field_vecs.append(emb) # [B, H]
            
        # concat fields → [B, H * num_fields]
        text_repr = torch.cat(field_vecs, dim=1)     
        return text_repr




class GatingLayer(nn.Module):
    def __init__(self, input_dim):
        super(GatingLayer, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tab_embedding, text_embedding):
        fused_input = torch.cat([tab_embedding, text_embedding], dim=1)
        g = self.gate(fused_input)  # shape: [batch_size, 1]
        gated_output = g * tab_embedding + (1 - g) * text_embedding
        return gated_output

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return self.block(x) + x  # Residual connection




class FusionModel(nn.Module):
    def __init__(self, tab_backbone: nn.Module, text_encoder: MultiTextEncoder, fusion_hidden: int = 256, out_dim: int = 1):
        super().__init__()
        self.tab_backbone = tab_backbone
        self.text_encoder = text_encoder
        tab_dim = getattr(tab_backbone, "output_dim")
        text_dim = len(text_encoder.field_names) * text_encoder.proj_dim


        self.head = nn.Sequential(
            #nn.Linear(tab_dim + text_dim, fusion_hidden),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, out_dim)
        )
        self.residual_tab = ResidualBlock(dim=128)
        self.residual_text = ResidualBlock(dim=128)
        

        self.g = GatingLayer(256)

    
    def forward(self, cats: torch.Tensor,conts: torch.Tensor, batch_texts: Dict[str, List[Union[List[str], str]]]) -> torch.Tensor:    
    
        tab_emb = self.tab_backbone(cats,conts)
        text_emb = self.text_encoder(batch_texts)
        tab_emb = self.residual_tab(tab_emb)
        text_emb = self.residual_tab(text_emb)
        gated = self.g(tab_emb,text_emb)
        return self.head(gated).squeeze(-1)


def concat_first_n(x,n=30 ,fixed_width=False):
    if not x:
        return "__MISSING__"
    if fixed_width:
        # exactly 10 per item, padded with spaces
        s = " ".join(((s or "")[:n].ljust(n) for s in x))
        return s.rstrip()[:200]
    else:
        # up to 10 per item, no padding; skip empty/None
        s = " ".join((s[:n] for s in x if s))
        return s.strip()[:200]

def safe_join(x):
    import numpy as np, ast
    n=35
    if isinstance(x, pd.Series):
        x = x.iloc[0]
        
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "__MISSING__"

    if isinstance(x, list):  

        x = concat_first_n(x,n=n, fixed_width=False)
        return x[:200]              
        #return " ".join(x).strip()[0:300] if len(x) > 0 else "__MISSING__"

    if isinstance(x, str):
        # Clean up the string before parsing
        cleaned = x.replace("\n", " ").replace("  ", " ").strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
                if isinstance(parsed, list):
                    parsed  = concat_first_n(parsed,n=n, fixed_width=False)
                    return parsed[:200]                            
                    #return " ".join(str(item).strip() for item in parsed if item).strip()[0:300]
            except Exception:
                pass
               
        return cleaned[0:200]

    return str(x).strip()[0:200]    
# ---------------------------
# Dataset
# ---------------------------
class TabTextDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, text_cols, target_col):
        self.df = df.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.text_cols = text_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cats = torch.tensor([row[c] for c in self.cat_cols], dtype=torch.long).to(device)
        conts = torch.tensor([row[c] for c in self.cont_cols], dtype=torch.float).to(device)

        texts = {}
        texts = {col: safe_join(row[col]) for col in self.text_cols}    
        target = torch.tensor(row[self.target_col], dtype=torch.float).to(device)
        #print(texts)
        return {"cats": cats, "conts": conts, "texts": texts, "target": target}
# ---------------------------
# Collate function
# ---------------------------
def collate_fn(batch):
    
    cats = torch.stack([b["cats"] for b in batch])
    conts = torch.stack([b["conts"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    
    texts = {}
    for field in batch[0]["texts"].keys():
        texts[field] = [b["texts"][field] for b in batch]

    return {"cats": cats, "conts": conts, "texts": texts, "target": targets}

# ---------------------------
# Training loop
# ---------------------------
import time
def train_epoch(model, dataloader, optimizer,device):
    model.train()
    running_loss = 0
    criterion =  nn.BCEWithLogitsLoss()
    i = 0
    for batch in dataloader:

        texts = {k: v for k, v in batch["texts"].items()}
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(batch['cats'],batch['conts'],texts)
        loss = criterion(logits,  batch["target"].to(device))  # nn.BCEWithLogitsLoss()

        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()  
        running_loss += loss_value
        
        if i%20 == 0:
            timestamp = time.strftime("%H:%M:%S")
            torch.cuda.empty_cache()
            print(f"\rBatch: {i} | GPU Memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB", end='', flush=True)

        i = i+1

            

    return running_loss / len(dataloader)


      
if __name__ == "__main__":
    # Example DataFrame with proper structure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #cat_cols = ['a_currency', 'a_product_interest','a_leads_recent']
    cont_cols = ['page_views', 'days_open']
    text_cols = ['processed_url', 'a_emailsubject']
    target    = ["a_is_won_float"]



    cat_cols = ['a_search_terms','a_product_interest','a_campaign_name','a_Type','a_leads_recent','a_currency']

    
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ['TRANSFORMERS_CACHE'] = "./hf_cache_smol/"
    model_name =  "HuggingFaceTB/SmolLM2-135M"


    
    #base = "./hf_cache/models--xlm-roberta-base/snapshots/"
    dataset = TabTextDataset(df_set, cat_cols, cont_cols, text_cols, target[0])
    loader = DataLoader(dataset, batch_size=64, shuffle=True,collate_fn=collate_fn)

    tab_backbone = CustomTabTransformer(categories=categories,num_continuous=len(cont_cols), output_dim=128).to(device)    
   
    model_txt = MultiTextEncoder(model_name, text_cols).to(device)
    model = FusionModel(tab_backbone=tab_backbone,text_encoder=model_txt).to(device) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-3)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=3e-2) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for i in range(20):
        loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Epoch Loss: {loss:.4f}  Epoch #: {i}")

    timestamp = time.strftime("%H:%M:%S")
    torch.save(model, f"./models/fused_model_{timestamp}.pth")
  
    batch_size = 512
    #dataset = TabTextDataset(df_set, cat_cols, cont_cols, text_cols, target[0])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    err = 0
    number_correct_convert = 0
    number_total_total = 0
    number_missed_convert = 0
    number_bad_convert = 0
    missed_convert = []
    bad_convert_pred = []
    
    with open("results.txt", "w") as f:
      with torch.no_grad():
        for batch in loader:
            
            targets = batch["target"].to(device)
            texts = {k: v for k, v in batch["texts"].items()}
            pred = model(batch['cats'],batch['conts'],texts)
            #pred = model(x_categ, x_cont, tokens, token_lens,False)
    
            
            for p,l in zip(torch.sigmoid(pred),targets):
    
    
    
              pp = float(p.squeeze().cpu().item())
              ll = float(l.squeeze().cpu().item())
    
              choice = pp >= 0.45
    
    
              f.write(f"Lead = {ll}, convert probability = {pp},  convert = {choice}\n")
    
              if choice == True and pp > 0.5:
                number_correct_convert = number_correct_convert + 1
    
              if ll > 0.5:
                number_total_total = number_total_total + 1
    
              if choice == True and  ll < 0.5:
                number_missed_convert = number_missed_convert + 1
                missed_convert.append(pp)
    
              if choice == False and  ll > 0.5:
                number_bad_convert = number_bad_convert + 1
                bad_convert_pred.append(pp)
    
    
              err = err + abs(pp - ll)
            break
        print(err/batch_size)
    
    print(f"number correct convert : {number_correct_convert}")
    print(f"number total convert : {number_total_total}")
    print('                              ')
    print('                              ')
    print(f"number bad convert : {number_missed_convert}")
    for mc in missed_convert:
      print(f"missed bad pred val : {mc}")
    
    
    print('                              ')
    print('                              ')
    
    print(f"number missed convert : {number_bad_convert}")
    for bc in bad_convert_pred:
      print(f"missed convert pred val : {bc}")


    #--------------------------------------------------------------------------------
    #import json
"""    
    def save_checkpoint(model, tokenizer, save_dir, extra_meta=None, pin_revision=None):
        os.makedirs(save_dir, exist_ok=True)
    
        # 1) Save minimal config to reconstruct the fused architecture
        cfg = {
            "hf_model_name": getattr(model, "hf_model_name", None),
            "arch_kwargs": getattr(model, "arch_kwargs", {}),
            "hf_revision": pin_revision,  # e.g., "main" or a commit hash
            "frozen_hf": True,            # reminder to re-freeze on load
        }
    
        # 2) Save the full state dict (includes your head and HF backbone)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": cfg,
                "meta": extra_meta or {},
                "torch_version": torch.__version__,
            },
            os.path.join(save_dir, "checkpoint.pt"),
        )
    
        # 3) Strongly recommended: save tokenizer (captures added vocab & specials)
        if tokenizer is not None:
            tok_dir = os.path.join(save_dir, "tokenizer")
            tokenizer.save_pretrained(tok_dir)

    save_all(model,None,None,None,f"./models/)            

"""
# OLD CODE #
"""
class MultiQueryPooling___(nn.Module):
    def __init__(self, hidden_size: int, num_queries: int = POOL_NUM_QUERIES, out_dim: int = POOL_OUT_DIM):
        super().__init__()
        
        self.hidden_size = hidden_size
        #self.query = nn.Parameter(torch.randn(hidden_dim))
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02) # [Q, H]
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.val_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self,hidden_states, attention_mask):
        
        B, N, H = hidden_states.shape
        K = self.key_proj(hidden_states) # [B, N, H]
        V = self.val_proj(hidden_states) # [B, N, H]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1) # [B, Q, H]

        # scores: [B, Q, N]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (H ** 0.5)
        mask = attention_mask.unsqueeze(1) # [B, 1, N]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1) # [B, Q, N]
        pooled = torch.matmul(attn, V) # [B, Q, H]
        pooled_flat = pooled.reshape(B, -1) # [B, Q*H]
        return pooled_flat

def tokenize_into_token_chunks____(text, tokenizer, max_len, max_chunks, max_total_tokens, add_special_tokens=False):

    tok = tokenizer.encode(
        text,
        add_special_tokens=True,   # usually required for model inference
        truncation=True,
        max_length=max_total_tokens             # counts special tokens automatically
    )
    
    tok = tok[:max_total_tokens] # clip to total tokens allowed
    chunks = []
    for i in range(0, len(tok), max_len):
        if len(chunks) >= max_chunks:
            break
        c = tok[i:i+max_len]
        chunks.append({"input_ids": c, "attention_mask": [1]*len(c)})

    if len(chunks) == 0:
        chunks.append({"input_ids": [], "attention_mask": []})
    return chunks


# --- Fixed MultiTextEncoder ---
class MultiTextEncoder___(nn.Module):
    
     def __init__(self, model_name, text_fields):
        super().__init__()

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ['TRANSFORMERS_CACHE'] = "./hf_cache_smol/"
        model_name =  "./hf_cache_smol/"  
        self.field_names = text_fields
        self.proj_dim = 512
       
        # Parameters you had
        self.MAX_TOKENS_PER_SAMPLE = 64*8
        self.CHUNK_LEN = 16
        self.MAX_CHUNKS = (self.MAX_TOKENS_PER_SAMPLE + self.CHUNK_LEN - 1) // self.CHUNK_LEN # -> 8

        self.text_fields = text_fields

        # load tokenizer and model (local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = tokenizer
        tokenizer.model_max_length = self.MAX_TOKENS_PER_SAMPLE

        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        model.config.use_cache = False

        # ensure pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

                
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        tokenizer.padding_side = "right"

        # Only needed if you *added* tokens, not if you reuse eos as pad
        model.resize_token_embeddings(len(tokenizer))

        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False

        self.transformer = model
        hidden_dim = self.transformer.config.hidden_size

        self.poolers = nn.ModuleDict({
            field: MultiQueryPooling(hidden_dim,1,hidden_dim) for field in text_fields
        })

        self.tab_proj = nn.Linear(hidden_dim , 256)   #should BN or add dropout? 128=dim_out
        self.tab_relu = nn.ReLU()
        self.tab_norm = nn.LayerNorm(256)
        

    def _encode_field_batch(self, texts, field_name):

        device = next(self.parameters()).device
        B = len(texts)

        # Step 1: tokenize each sample into chunks, keep track of counts
        all_chunk_items = [] # list of small dicts acceptable to tokenizer.pad
        chunks_per_sample = [] # how many chunks each sample produced
        for text in texts:
            chunks = tokenize_into_token_chunks(
                text,
                self.tokenizer,
                max_len=self.CHUNK_LEN,
                max_chunks=self.MAX_CHUNKS,
                max_total_tokens=self.MAX_TOKENS_PER_SAMPLE,
                add_special_tokens=False
            )
            # normalize each chunk item to contain keys tokenizer.pad expects
            for c in chunks:
                # ensure lists (input_ids, attention_mask) exist
                if "input_ids" not in c:
                    c["input_ids"] = []
                if "attention_mask" not in c:
                    c["attention_mask"] = [1] * len(c["input_ids"])
                all_chunk_items.append(c)
            chunks_per_sample.append(len(chunks))

        total_chunks = len(all_chunk_items)
        if total_chunks == 0:
            # fallback: produce zeros
            H = self.transformer.config.hidden_size
            return torch.zeros(B, self.MAX_CHUNKS, H, device=device), torch.zeros(B, self.MAX_CHUNKS, dtype=torch.long, device=device)

        # Step 2: pad all chunk items together into a single batch for the transformer
        padded = self.tokenizer.pad(
            all_chunk_items,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"].to(device) # [total_chunks, Lmax]
        attention_mask = padded["attention_mask"].to(device) # [total_chunks, Lmax]

        # Step 3: run transformer on flattened chunks
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        # last hidden state: outputs.hidden_states[-1] or outputs.last_hidden_state (depending on model)
        # Using hidden_states[-1] to match your code:
        if outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1] # [total_chunks, seq_len, H]
        else:
            last_hidden_state = outputs.last_hidden_state # fallback

        # Step 4: pool each chunk to vector
        pooler = self.poolers[field_name]
        chunk_embs = pooler(last_hidden_state, attention_mask) # [total_chunks, H]

        # Step 5: split chunk_embs back into per-sample lists
        H = chunk_embs.size(-1)
        per_sample_chunks = []
        idx = 0
        max_chunks_observed = 0
        for n in chunks_per_sample:
            if n == 0:
                per_sample_chunks.append(torch.zeros(0, H, device=device))
            else:
                per_sample_chunks.append(chunk_embs[idx: idx + n]) # [n, H]
            idx += n
            if n > max_chunks_observed:
                max_chunks_observed = n

        # Step 6: pad per-sample chunk sequences up to max_chunks_observed (or self.MAX_CHUNKS)
        target_chunks = min(max(max_chunks_observed, 1), self.MAX_CHUNKS) # avoid zero, limit to MAX_CHUNKS
        padded_chunk_tensors = []
        chunk_masks = []
        for pe in per_sample_chunks:
            n = pe.size(0)
            if n == 0:
                pad_tensor = torch.zeros(target_chunks, H, device=device)
                mask = torch.zeros(target_chunks, device=device, dtype=torch.long)
                padded_chunk_tensors.append(pad_tensor)
                chunk_masks.append(mask)
            else:
                if n < target_chunks:
                    pad_needed = target_chunks - n
                    pad_tensor = torch.cat([pe, torch.zeros(pad_needed, H, device=device)], dim=0) # [target_chunks, H]
                    mask = torch.cat([torch.ones(n, device=device, dtype=torch.long),
                                      torch.zeros(pad_needed, device=device, dtype=torch.long)])
                else:
                    pad_tensor = pe[:target_chunks] # truncate if necessary
                    mask = torch.ones(target_chunks, device=device, dtype=torch.long)
                padded_chunk_tensors.append(pad_tensor)
                chunk_masks.append(mask)

        # stack to [B, target_chunks, H] and mask [B, target_chunks]
        chunk_embeddings = torch.stack(padded_chunk_tensors, dim=0)
        chunk_mask = torch.stack(chunk_masks, dim=0)

        return chunk_embeddings, chunk_mask


    def forward(self, texts_dict):
        outputs = {}
        for field in self.text_fields:
            emb, mask = self._encode_field_batch(texts_dict[field], field)
            outputs[field] = (emb, mask)
    
        # --- NEW: reduce each [B, C, H] → [B, H] ---
        field_vecs = []
        for field, (emb, mask) in outputs.items():
            # simple mean across chunks
            emb = self.tab_proj(emb.mean(dim=1))
            emb = self.tab_relu(emb)
            emb = self.tab_norm(emb)
            field_vecs.append(emb) # [B, H]
            
    
        # concat fields → [B, H * num_fields]
        text_repr = torch.cat(field_vecs, dim=1)

        
        return text_repr


        
"""        

    


          



