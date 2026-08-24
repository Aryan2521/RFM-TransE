import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    """
    Standard TransE model.
    The translation hypothesis states that: head + relation \approx tail.
    
    Provides the original fixed post-hoc mapping as an illustrative reference.
    """
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=2.0, p_norm=1):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights using the uniform distribution:
        [-6 / sqrt(d), 6 / sqrt(d)]
        """
        bound = 6.0 / (self.embedding_dim ** 0.5)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
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
        """
        h_emb = self.entity_embeddings(heads)
        r_emb = self.relation_embeddings(relations)
        t_emb = self.entity_embeddings(tails)

        if self.p_norm == 1:
            distance = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
        elif self.p_norm == 2:
            distance = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
        else:
            raise ValueError(f"p_norm must be 1 or 2, got {self.p_norm}")
            
        return distance

    def fuzzy_score(self, heads, relations, tails, alpha_default=0.1):
        """
        Fixed post-hoc sigmoid retained only as an illustrative reference.

        This mapping cannot exceed 0.5 because TransE distances are
        non-negative.
        """
        distance = self.forward(heads, relations, tails)
        return torch.sigmoid(-alpha_default * distance)

    def get_loss(self, pos_distances, neg_distances):
        """
        Compute standard MarginRankingLoss.
        """
        target = torch.tensor([-1.0], device=pos_distances.device)
        loss_fn = nn.MarginRankingLoss(margin=self.margin, reduction='mean')
        return loss_fn(pos_distances, neg_distances, target)

    def train_step(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t, pos_labels, neg_labels, optimizer, rules=None):
        """
        Execute standard TransE optimization step. Only trains on translation distances.
        """
        optimizer.zero_grad()
        
        pos_distances = self.forward(pos_h, pos_r, pos_t)
        neg_distances = self.forward(neg_h, neg_r, neg_t)
        
        loss = self.get_loss(pos_distances, neg_distances)
        
        loss.backward()
        optimizer.step()
        self.normalize_entities()
        
        return loss.item()
