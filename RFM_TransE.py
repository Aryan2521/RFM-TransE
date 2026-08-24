import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RFMTransE(nn.Module):
    """
    RFM-TransE: A Relational Fuzzy Membership Knowledge Graph Embedding Model.
    Version: Core formulation (without logical rule regularization).
    
    1. Fuzzy truth is the central membership function: 
       truth = Sigmoid(beta_rel * (gamma_rel - distance) + fuzzy_bias)
    2. Learnable relation-specific fuzzy parameters beta (sharpness) and gamma (base tolerance).
    3. Strictly positive constraints applied via PyTorch F.softplus().
    4. Logits-based outputs to support stable BCEWithLogitsLoss optimization.
    5. Clean device and type safety.
    """
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=2.0, p_norm=1, 
                 alpha=0.1, lambda_fuzzy=0.1, fuzzy_bias=0.0,
                 membership_function="sigmoid"):
        super(RFMTransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm
        
        self.lambda_fuzzy = lambda_fuzzy
        
        assert membership_function in ["exponential", "sigmoid", "gaussian"], \
            "membership_function must be 'exponential', 'sigmoid', or 'gaussian'"
        self.membership_function = membership_function
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Learnable bias for sigmoid membership function
        self.fuzzy_bias = nn.Parameter(torch.tensor(float(fuzzy_bias)))
        
        # Initialize relation-specific parameters using inverse softplus
        # Effective beta begins around initial alpha, and base gamma starts near margin / 2.
        # Because fuzzy_bias is initialized to zero, gamma is also the initial 0.5 transition distance.
        beta_init_target = max(alpha - 1e-4, 1e-6)
        raw_beta_init = np.log(np.exp(beta_init_target) - 1.0)
        
        gamma_init_target = max(margin / 2.0 - 1e-4, 1e-6)
        raw_gamma_init = np.log(np.exp(gamma_init_target) - 1.0)
        
        self.raw_beta_rel = nn.Parameter(torch.full((num_relations,), float(raw_beta_init)))
        self.raw_gamma_rel = nn.Parameter(torch.full((num_relations,), float(raw_gamma_init)))
        
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights using the uniform distribution:
        [-6 / sqrt(d), 6 / sqrt(d)]
        """
        bound = 6.0 / (self.embedding_dim ** 0.5)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        
        # Normalize entity embeddings to unit norm initially
        self.normalize_entities()

    def normalize_entities(self):
        """
        Normalize entity embeddings to unit L2 norm.
        """
        with torch.no_grad():
            self.entity_embeddings.weight.copy_(
                F.normalize(self.entity_embeddings.weight, p=2, dim=1)
            )

    def forward(self, heads, relations, tails):
        """
        Compute the translation distance for a batch of triples.
        heads, relations, tails: LongTensors of shape (batch_size,)
        Returns a distance score tensor of shape (batch_size,)
        """
        h_emb = self.entity_embeddings(heads)
        r_emb = self.relation_embeddings(relations)
        t_emb = self.entity_embeddings(tails)
        
        # Distance: ||h + r - t||_p
        if self.p_norm == 1:
            distance = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
        elif self.p_norm == 2:
            distance = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
        else:
            raise ValueError(f"p_norm must be 1 or 2, got {self.p_norm}")
            
        return distance

    def fuzzy_logits(self, heads, relations, tails):
        """
        Compute the fuzzy logit value before applying sigmoid.
        Allows numerically stable training using BCEWithLogitsLoss.
        Only valid when membership_function is "sigmoid".
        
        Formula:
          logits = beta_rel * (gamma_rel - distance) + fuzzy_bias

        Here gamma_rel is a base tolerance. The effective distance at which
        sigmoid membership equals 0.5 is gamma_rel + fuzzy_bias / beta_rel.
        """
        if self.membership_function != "sigmoid":
            raise ValueError("fuzzy_logits is only supported for the 'sigmoid' membership function.")
            
        distance = self.forward(heads, relations, tails)
        
        # Transform raw parameters via softplus to enforce strict positivity
        beta_val = F.softplus(self.raw_beta_rel[relations]) + 1e-4
        gamma_val = F.softplus(self.raw_gamma_rel[relations]) + 1e-4
        
        logits = beta_val * (gamma_val - distance) + self.fuzzy_bias
        return logits

    def fuzzy_score(self, heads, relations, tails):
        """
        Compute the fuzzy truth/confidence score of triples (membership value in [0, 1]).
        
        Formulas:
          - Sigmoid: Sigmoid(beta_rel * (gamma_rel - distance) + fuzzy_bias)
          - Exponential: exp(-beta_rel * max(0, distance - gamma_rel))
          - Gaussian: exp(-beta_rel * (max(0, distance - gamma_rel))^2)
        """
        distance = self.forward(heads, relations, tails)
        
        # Transform raw parameters via softplus to enforce strict positivity
        beta_val = F.softplus(self.raw_beta_rel[relations]) + 1e-4
        gamma_val = F.softplus(self.raw_gamma_rel[relations]) + 1e-4
        
        if self.membership_function == "sigmoid":
            logits = beta_val * (gamma_val - distance) + self.fuzzy_bias
            return torch.sigmoid(logits)
        elif self.membership_function == "exponential":
            return torch.exp(-beta_val * torch.clamp(distance - gamma_val, min=0.0))
        elif self.membership_function == "gaussian":
            return torch.exp(-beta_val * (torch.clamp(distance - gamma_val, min=0.0) ** 2))
        else:
            raise ValueError(f"Unknown membership function: {self.membership_function}")

    # --- Loss Methods ---
    def get_loss(self, pos_distances, neg_distances):
        """
        Compute the margin-based ranking loss for TransE.
        """
        target = torch.tensor([-1.0], device=pos_distances.device)
        loss_fn = nn.MarginRankingLoss(margin=self.margin, reduction='mean')
        return loss_fn(pos_distances, neg_distances, target)

    def get_fuzzy_loss(self, predicted_val, target_labels):
        """
        Compute fuzzy loss.
        For sigmoid, uses BCEWithLogitsLoss on logits for numerical stability.
        For exponential and gaussian, uses BCELoss with clamped predictions.
        """
        if self.membership_function == "sigmoid":
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            return loss_fn(predicted_val, target_labels)
        else:
            predicted_truth = torch.clamp(predicted_val, min=1e-7, max=1.0 - 1e-7)
            loss_fn = nn.BCELoss(reduction='mean')
            return loss_fn(predicted_truth, target_labels)

    def get_combined_loss(self, pos_distances, neg_distances, pos_pred, neg_pred, 
                           pos_labels, neg_labels):
        """
        Compute total loss: ranking_loss + lambda_fuzzy * fuzzy_loss
        """
        ranking_loss = self.get_loss(pos_distances, neg_distances)
        fuzzy_loss = self.get_fuzzy_loss(pos_pred, pos_labels) + self.get_fuzzy_loss(neg_pred, neg_labels)
        return ranking_loss + self.lambda_fuzzy * fuzzy_loss

    def train_step(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t, pos_labels, neg_labels, optimizer, rules=None):
        """
        Execute a single optimization step.
        """
        optimizer.zero_grad()
        
        pos_distances = self.forward(pos_h, pos_r, pos_t)
        neg_distances = self.forward(neg_h, neg_r, neg_t)
        
        if self.membership_function == "sigmoid":
            pos_pred = self.fuzzy_logits(pos_h, pos_r, pos_t)
            neg_pred = self.fuzzy_logits(neg_h, neg_r, neg_t)
        else:
            pos_pred = self.fuzzy_score(pos_h, pos_r, pos_t)
            neg_pred = self.fuzzy_score(neg_h, neg_r, neg_t)
            
        loss = self.get_combined_loss(
            pos_distances, neg_distances, pos_pred, neg_pred, 
            pos_labels, neg_labels
        )
        
        loss.backward()
        optimizer.step()
        self.normalize_entities()
        
        return loss.item()
