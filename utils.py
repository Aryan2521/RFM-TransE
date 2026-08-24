import os
import urllib.request
import random
import ssl
import torch
import numpy as np

STRUCTURAL_FUZZY_POSITIVE_BASES = {
    "low": 0.95,
    "medium": 0.80,
    "high": 0.60,
}


def get_structural_fuzzy_base(uncertainty_level, is_positive):
    """Return the unadjusted target used by structural fuzzy labeling."""
    try:
        base_pos = STRUCTURAL_FUZZY_POSITIVE_BASES[uncertainty_level]
    except KeyError as exc:
        raise ValueError(f"Unknown uncertainty level: {uncertainty_level}") from exc
    return base_pos if is_positive else 0.0

# Workaround for macOS Python environment missing SSL certificates
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

class RealKG:
    """
    Downloads, parses, and caches Knowledge Graph datasets (FB15k, FB15k-237, WN18RR, CoDEx-S, CoDEx-M, YAGO3-10).
    """
    URL_TEMPLATES = {
        "FB15k": {
            "train": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k/train.txt",
            "valid": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k/valid.txt",
            "test": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k/test.txt"
        },
        "FB15k-237": {
            "train": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/train.txt",
            "valid": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/valid.txt",
            "test": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/test.txt"
        },
        "WN18RR": {
            "train": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/train.txt",
            "valid": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/valid.txt",
            "test": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/test.txt"
        },
        "CoDEx-S": {
            "train": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/train.txt",
            "valid": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/valid.txt",
            "test": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/test.txt"
        },
        "CoDEx-M": {
            "train": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/train.txt",
            "valid": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/valid.txt",
            "test": "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/test.txt"
        },
        "YAGO3-10": {
            "train": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
            "valid": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
            "test": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt"
        }
    }
    
    def __init__(self, dataset_name, data_dir="data"):
        if dataset_name not in self.URL_TEMPLATES:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Choose 'FB15k', 'FB15k-237', 'WN18RR', 'CoDEx-S', 'CoDEx-M', or 'YAGO3-10'.")
            
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(data_dir, dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        
        self.download_dataset()
        self.load_dataset()

    def download_dataset(self):
        urls = self.URL_TEMPLATES[self.dataset_name]
        for split, url in urls.items():
            filename = f"{split}.txt"
            filepath = os.path.join(self.dataset_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Downloading {self.dataset_name} {split} set from {url}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"Saved to {filepath}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    raise e
            else:
                print(f"Found cached {self.dataset_name} {split} set at {filepath}")

    def load_dataset(self):
        splits = ["train", "valid", "test"]
        raw_triples = {split: [] for split in splits}
        
        # Pass 1: Build entity and relation vocabularies
        for split in splits:
            filepath = os.path.join(self.dataset_dir, f"{split}.txt")
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        parts = line.strip().split() # Fallback to any whitespace split
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        raw_triples[split].append((h, r, t))
                        
                        # Add to vocab if not present
                        if h not in self.entity_to_id:
                            eid = len(self.entity_to_id)
                            self.entity_to_id[h] = eid
                            self.id_to_entity[eid] = h
                        if t not in self.entity_to_id:
                            eid = len(self.entity_to_id)
                            self.entity_to_id[t] = eid
                            self.id_to_entity[eid] = t
                        if r not in self.relation_to_id:
                            rid = len(self.relation_to_id)
                            self.relation_to_id[r] = rid
                            self.id_to_relation[rid] = r

        # Pass 2: Convert to ID triples
        self.train_triples = self.convert_to_ids(raw_triples["train"])
        self.val_triples = self.convert_to_ids(raw_triples["valid"])
        self.test_triples = self.convert_to_ids(raw_triples["test"])

    def convert_to_ids(self, raw_list):
        id_list = []
        for h, r, t in raw_list:
            h_id = self.entity_to_id[h]
            r_id = self.relation_to_id[r]
            t_id = self.entity_to_id[t]
            id_list.append((h_id, r_id, t_id))
        return id_list

    def get_corrupted_triples(self, triples, all_triples_set, num_entities):
        """
        Generate negative corrupted triples (1:1 ratio) for triple classification.
        For each triple, corrupt either head or tail with a random entity.
        Ensures the corrupted triple does not exist in all_triples_set.
        """
        corrupted_triples = []
        for h, r, t in triples:
            while True:
                if random.random() < 0.5:
                    h_prime = random.randint(0, num_entities - 1)
                    neg_triple = (h_prime, r, t)
                else:
                    t_prime = random.randint(0, num_entities - 1)
                    neg_triple = (h, r, t_prime)
                
                if neg_triple not in all_triples_set:
                    corrupted_triples.append(neg_triple)
                    break
        return corrupted_triples

def generate_training_negatives(pos_triples, num_entities):
    neg_triples = []
    for h, r, t in pos_triples:
        if random.random() < 0.5:
            h_prime = random.randint(0, num_entities - 1)
            neg_triples.append((h_prime, r, t))
        else:
            t_prime = random.randint(0, num_entities - 1)
            neg_triples.append((h, r, t_prime))
    return neg_triples

def generate_fuzzy_labels(triples_or_size, is_positive, uncertainty_level="low", device="cpu", entity_neighbors=None):
    r"""
    Generates synthetic fuzzy truth labels representing uncertainty.
    
    If entity_neighbors (adjacency list mapping entity_id to set of neighbor_ids) is provided 
    and a list of triples is passed, computes structural fuzzy labels based on the Jaccard similarity 
    between head and tail neighborhoods:
      J(h, t) = |N(h) \cap N(t)| / |N(h) \cup N(t)|
      y_pos = base_pos + (1 - base_pos) * J(h, t)
      y_neg = (1 - base_pos) * J(h', t)
    
    Otherwise (or if triples_or_size is an integer), falls back to generating flat fuzzy labels.
    """
    if isinstance(triples_or_size, int):
        size = triples_or_size
        triples = None
    else:
        triples = triples_or_size
        size = len(triples)

    # 1. Structural Fuzzy Labeling using Jaccard Similarity of neighborhoods
    if entity_neighbors is not None and triples is not None:
        base_pos = get_structural_fuzzy_base(uncertainty_level, True)
            
        labels_list = []
        for triple in triples:
            h, r, t = triple[0], triple[1], triple[2]
            
            # Handle tensor/numpy conversions if needed
            if hasattr(h, 'item'):
                h = h.item()
            if hasattr(t, 'item'):
                t = t.item()
                
            N_h = entity_neighbors.get(h, set())
            N_t = entity_neighbors.get(t, set())
            
            union_len = len(N_h.union(N_t))
            if union_len > 0:
                jaccard = len(N_h.intersection(N_t)) / union_len
            else:
                jaccard = 0.0
                
            if is_positive:
                # Positive triples: base_pos + (1 - base_pos) * Jaccard
                val = base_pos + (1.0 - base_pos) * jaccard
            else:
                # Negative triples: (1 - base_pos) * Jaccard
                val = (1.0 - base_pos) * jaccard
                
            labels_list.append(val)
            
        return torch.tensor(labels_list, dtype=torch.float, device=device)

    # 2. Fallback Flat Labeling
    if uncertainty_level == "low":
        pos_val, neg_val, noise_prob = 0.95, 0.05, 0.0
    elif uncertainty_level == "medium":
        pos_val, neg_val, noise_prob = 0.80, 0.20, 0.15
    elif uncertainty_level == "high":
        pos_val, neg_val, noise_prob = 0.60, 0.40, 0.35
    else:
        raise ValueError(f"Unknown uncertainty level: {uncertainty_level}")
        
    base_val = pos_val if is_positive else neg_val
    labels = torch.full((size,), base_val, dtype=torch.float, device=device)
    
    if noise_prob > 0.0:
        mask = torch.rand(size, device=device) < noise_prob
        labels[mask] = torch.rand(mask.sum(), device=device)
        
    return labels

def evaluate_link_prediction(model, test_triples, all_triples_set, num_entities, device, batch_size=64):
    """
    Evaluates Link Prediction (Ranking metrics: MR, MRR, Hits@3, Hits@5).
    Uses the 'filtered' setting to avoid penalizing true positive facts.
    Vectorized using PyTorch broadcasting for high performance.
    """
    model.eval()
    ranks = []
    
    with torch.no_grad():
        E_emb = model.entity_embeddings.weight # shape [num_entities, D]
        num_triples = len(test_triples)
        
        pos_tails = {}
        pos_heads = {}
        for h, r, t in all_triples_set:
            if (h, r) not in pos_tails:
                pos_tails[(h, r)] = []
            pos_tails[(h, r)].append(t)
            
            if (r, t) not in pos_heads:
                pos_heads[(r, t)] = []
            pos_heads[(r, t)].append(h)
        
        for i in range(0, num_triples, batch_size):
            batch = test_triples[i : i + batch_size]
            B = len(batch)
            
            heads = [t[0] for t in batch]
            relations = [t[1] for t in batch]
            tails = [t[2] for t in batch]
            
            h_ids = torch.tensor(heads, dtype=torch.long, device=device)
            r_ids = torch.tensor(relations, dtype=torch.long, device=device)
            t_ids = torch.tensor(tails, dtype=torch.long, device=device)
            
            h_emb = model.entity_embeddings(h_ids) # shape [B, D]
            r_emb = model.relation_embeddings(r_ids) # shape [B, D]
            t_emb = model.entity_embeddings(t_ids) # shape [B, D]
            
            # --- 1. Tail Corruption: (h, r, ?) ---
            h_plus_r = h_emb + r_emb # shape [B, D]
            diff_tail = h_plus_r.unsqueeze(1) - E_emb.unsqueeze(0)
            
            if model.p_norm == 1:
                tail_distances = torch.sum(torch.abs(diff_tail), dim=-1) # shape [B, num_entities]
            else:
                tail_distances = torch.norm(diff_tail, p=2, dim=-1) # shape [B, num_entities]
                
            target_tail_dists = tail_distances[torch.arange(B), t_ids] # shape [B]
            
            tail_distances_cpu = tail_distances.cpu().numpy()
            target_tail_dists_cpu = target_tail_dists.cpu().numpy()
            
            for j in range(B):
                h_j, r_j, t_j = batch[j]
                for t_prime in pos_tails.get((h_j, r_j), []):
                    if t_prime != t_j:
                        tail_distances_cpu[j, t_prime] = np.inf
                        
                tail_rank = np.sum(tail_distances_cpu[j] < target_tail_dists_cpu[j]) + 1
                ranks.append(tail_rank)
                
            # --- 2. Head Corruption: (?, r, t) ---
            r_minus_t = r_emb - t_emb # shape [B, D]
            diff_head = E_emb.unsqueeze(0) + r_minus_t.unsqueeze(1)
            
            if model.p_norm == 1:
                head_distances = torch.sum(torch.abs(diff_head), dim=-1) # shape [B, num_entities]
            else:
                head_distances = torch.norm(diff_head, p=2, dim=-1) # shape [B, num_entities]
                
            target_head_dists = head_distances[torch.arange(B), h_ids] # shape [B]
            
            head_distances_cpu = head_distances.cpu().numpy()
            target_head_dists_cpu = target_head_dists.cpu().numpy()
            
            for j in range(B):
                h_j, r_j, t_j = batch[j]
                for h_prime in pos_heads.get((r_j, t_j), []):
                    if h_prime != h_j:
                        head_distances_cpu[j, h_prime] = np.inf
                        
                head_rank = np.sum(head_distances_cpu[j] < target_head_dists_cpu[j]) + 1
                ranks.append(head_rank)
                
    ranks = np.array(ranks)
    mr = np.mean(ranks)
    mrr = np.mean(1.0 / ranks)
    hits3 = np.mean(ranks <= 3)
    hits5 = np.mean(ranks <= 5)
    
    return {
        "Mean Rank": mr,
        "Mean Reciprocal Rank": mrr,
        "Hits@3": hits3,
        "Hits@5": hits5
    }

def optimize_thresholds(model, val_pos, val_neg, num_relations, device):
    """
    Finds relation-specific distance thresholds that maximize the validation F1 score.
    Uses grid-search approximation for large datasets.
    """
    model.eval()
    
    val_pos_by_rel = {r: [] for r in range(num_relations)}
    val_neg_by_rel = {r: [] for r in range(num_relations)}
    
    with torch.no_grad():
        h_pos = torch.tensor([t[0] for t in val_pos], dtype=torch.long, device=device)
        r_pos = torch.tensor([t[1] for t in val_pos], dtype=torch.long, device=device)
        t_pos = torch.tensor([t[2] for t in val_pos], dtype=torch.long, device=device)
        if len(val_pos) > 0:
            pos_distances = []
            for start in range(0, len(val_pos), 5000):
                batch_pos_dists = model(
                    h_pos[start : start + 5000], 
                    r_pos[start : start + 5000], 
                    t_pos[start : start + 5000]
                ).cpu().numpy()
                pos_distances.extend(batch_pos_dists)
            
            for (h, r, t), dist in zip(val_pos, pos_distances):
                val_pos_by_rel[r].append(dist)
                
        h_neg = torch.tensor([t[0] for t in val_neg], dtype=torch.long, device=device)
        r_neg = torch.tensor([t[1] for t in val_neg], dtype=torch.long, device=device)
        t_neg = torch.tensor([t[2] for t in val_neg], dtype=torch.long, device=device)
        if len(val_neg) > 0:
            neg_distances = []
            for start in range(0, len(val_neg), 5000):
                batch_neg_dists = model(
                    h_neg[start : start + 5000], 
                    r_neg[start : start + 5000], 
                    t_neg[start : start + 5000]
                ).cpu().numpy()
                neg_distances.extend(batch_neg_dists)
                
            for (h, r, t), dist in zip(val_neg, neg_distances):
                val_neg_by_rel[r].append(dist)
                
    thresholds = {}
    
    for r in range(num_relations):
        pos_dists = np.array(val_pos_by_rel[r])
        neg_dists = np.array(val_neg_by_rel[r])
        
        if len(pos_dists) == 0 and len(neg_dists) == 0:
            thresholds[r] = 0.0
            continue
            
        all_dists = np.concatenate([pos_dists, neg_dists])
        
        if len(all_dists) > 200:
            candidate_thresholds = np.linspace(np.min(all_dists), np.max(all_dists), 200)
        else:
            candidate_thresholds = np.sort(all_dists)
            
        best_f1 = -1.0
        best_threshold = 0.0
        
        for thresh in candidate_thresholds:
            tp = np.sum(pos_dists <= thresh)
            fp = np.sum(neg_dists <= thresh)
            fn = np.sum(pos_dists > thresh)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
        thresholds[r] = best_threshold
        
    return thresholds

def evaluate_triple_classification(model, test_pos, test_neg, thresholds, device):
    """
    Evaluates Triple Classification (Precision, Recall, F1 score) on test set
    using relation-specific thresholds.
    """
    model.eval()
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    with torch.no_grad():
        if len(test_pos) > 0:
            h_pos = torch.tensor([t[0] for t in test_pos], dtype=torch.long, device=device)
            r_pos = torch.tensor([t[1] for t in test_pos], dtype=torch.long, device=device)
            t_pos = torch.tensor([t[2] for t in test_pos], dtype=torch.long, device=device)
            
            pos_distances = []
            for start in range(0, len(test_pos), 5000):
                batch_pos_dists = model(
                    h_pos[start : start + 5000], 
                    r_pos[start : start + 5000], 
                    t_pos[start : start + 5000]
                ).cpu().numpy()
                pos_distances.extend(batch_pos_dists)
            
            for (h, r, t), dist in zip(test_pos, pos_distances):
                thresh = thresholds.get(r, 0.0)
                if dist <= thresh:
                    tp += 1
                else:
                    fn += 1
                    
        if len(test_neg) > 0:
            h_neg = torch.tensor([t[0] for t in test_neg], dtype=torch.long, device=device)
            r_neg = torch.tensor([t[1] for t in test_neg], dtype=torch.long, device=device)
            t_neg = torch.tensor([t[2] for t in test_neg], dtype=torch.long, device=device)
            
            neg_distances = []
            for start in range(0, len(test_neg), 5000):
                batch_neg_dists = model(
                    h_neg[start : start + 5000], 
                    r_neg[start : start + 5000], 
                    t_neg[start : start + 5000]
                ).cpu().numpy()
                neg_distances.extend(batch_neg_dists)
            
            for (h, r, t), dist in zip(test_neg, neg_distances):
                thresh = thresholds.get(r, 0.0)
                if dist <= thresh:
                    fp += 1
                else:
                    tn += 1
                    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def compute_fuzzy_evaluation_metrics(pred_scores, target_labels, num_bins=10):
    """
    Computes MSE, MAE, and Expected Calibration Error (ECE) for fuzzy truth predictions.
    pred_scores: np.array of shape (N,) containing predicted fuzzy truth scores in [0, 1]
    target_labels: np.array of shape (N,) containing target soft labels in [0, 1]
    """
    pred_scores = np.clip(pred_scores, 0.0, 1.0)
    target_labels = np.clip(target_labels, 0.0, 1.0)
    
    mse = np.mean((pred_scores - target_labels) ** 2)
    mae = np.mean(np.abs(pred_scores - target_labels))
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        in_bin = (pred_scores >= bin_lower) & (pred_scores < bin_upper)
        if m == num_bins - 1:
            in_bin = in_bin | (pred_scores == bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_predicted = np.mean(pred_scores[in_bin])
            avg_target = np.mean(target_labels[in_bin])
            ece += prop_in_bin * np.abs(avg_predicted - avg_target)
            
    return {
        "MSE": mse,
        "MAE": mae,
        "ECE": ece
    }
