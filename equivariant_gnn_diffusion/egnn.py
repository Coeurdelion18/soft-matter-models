import torch
import torch.nn as nn

def mlp(in_dim, hidden_dim, out_dim, n_hidden=1, act=nn.SiLU):
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_hidden):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim=0, act=nn.SiLU):
        super().__init__()
        msg_in = 2 * hidden_dim + 1 + edge_feat_dim
        self.edge_mlp = mlp(msg_in, hidden_dim, hidden_dim, act=act)
        self.vector_gate = mlp(hidden_dim, hidden_dim, 1, act=act)
        self.node_mlp = mlp(hidden_dim + hidden_dim, hidden_dim, hidden_dim, act=act)
    #self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        """
        Note: self.vector_gate is a scalar quantity which is used to scale the equivariant vector.
        """

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        'h' is the node features, of dimension (N, F) where N is the number of nodes and F is the number of node features
        'x' is the node coordinates, of dimension (N, D) where D is the number of spatial dimensions
        'edge_index' is of shape (2, E) where 'E' is the number of edges. [0] is the source node list, [1] is the destination node list
        """
        src, dst = edge_index
        rel = x[src] - x[dst] #Dimension (E, 2), equivariant
        dist2 = (rel ** 2).sum(-1, keepdim=True) #Dimension (E, 1), invariant
        #Equivariant GNNs use dist2 as an input because it is invariant under rotation and translation

        m_in = [h[src], h[dst], dist2]

        if edge_attr is not None:
            m_in.append(edge_attr)

        m_in = torch.cat(m_in, dim=-1)
        #m_in will be of dimension (E, 2F + 1 + edge_feat_dim)
        m_ij = self.edge_mlp(m_in)
        #m_ij will be of dimension (E, hidden_dim) i.e. (E, F)
        #Note that throughout, F == hidden_dim, OR, h arrives as (N, hidden_dim) and not (N, node_feat_dim)
        #Normally, you could add an encoder layer between F and hidden_dim (h = self.node_encoder(h) at the beginning of the forward pass), but I'm going to keep it this way because this way it's one less hyperparameter to tune
        #Now we update the relative position with a learned scalar
        gate = self.vector_gate(m_ij) #Dimension (E, 1)
        #Basically, we're looking at all the edges and learning what scalar the relative vector should be multiplied by to be more accurate
        x_msg = gate * rel #Dimension (E, 2)
        #x_msg is the set of scaled relative vectors

        N = h.shape[0] #Number of nodes
        agg_m = torch.zeros(N, m_ij.shape[-1], device=h.device, dtype=h.dtype)
        #So agg_m is a zeros tensor of shape (N, F)
        #m_ij, remember, is of shape (E, F)
        #dst is of shape (E,)
        agg_m.index_add_(0, dst, m_ij)
        #index_add_ collects all messages arriving at the same destination node and stores it at agg_m[node]
        agg_m /= deg.clamp(min=1)

        agg_x = torch.zeros(N, 2, device=h.device, dtype=h.dtype)
        agg_x.index_add_(0, dst, x_msg)
        #Aggregates all the scaled relative vectors pointing to node 'j' at agg_x[j]

        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        #Aggregates the degree of node 'j' at deg[j]

        agg_x = agg_x / deg.clamp(min=1.0) #Because otherwise, nodes with a large in-degree move much more than those with a smaller one
        h_new = h + self.node_mlp(torch.cat([h, agg_m], dim=-1))
        #h is of shape (N, F)
        #agg_m is of shape (N, F)
        #Concatenating gives you a tensor of shape (N, 2F)
        #The MLP gives you back a tensor of shape (N, F) as defined by the out_dim of node_mlp

        x_new = x + agg_x

        return h_new, x_new
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim): #The dim here will correspond to the hidden_dim
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
 
    def forward(self, t):
        # t: (N, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=t.dtype) * (9.0 / max(half - 1, 1))
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], self.dim - emb.shape[-1], device=t.device, dtype=t.dtype)], dim=-1)
        return self.proj(emb)
"""
Regarding the apparent confusion between node_feat_dim and hidden_dim -
For this purely generative model (unconditional), the node features going into the first layer are just the sinusoidal time embedding (plus optional global conditioning). 
The time embedding is already output at hidden_dim size by design — SinusoidalEmbedding has a nn.Linear(dim, dim) projection as its final step, 
so it already lands in the right space.
So we don't need another encoder to map between the two dimensions
In the case of the conditional model for the cavitation problem, we have an encoder, but OUTSIDE the egnn layer (so h arrives as shape (N, hidden_dim) already)
"""
class EGNNUnconditionalDenoiser(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=4, edge_feat_dim=1, n_scalar_feats=2, global_cond_dim=0):
        super().__init__()
        self.global_cond_dim = global_cond_dim
        in_dim = 2 * hidden_dim + global_cond_dim  # time embedding + scalar projection
        self.input_mlp = mlp(in_dim, hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, edge_feat_dim=edge_feat_dim) for _ in range(n_layers)])

        self.time_embed = SinusoidalEmbedding(hidden_dim)
        self.scalar_proj = nn.Linear(n_scalar_feats, hidden_dim)  # size, type, and any other scalars
        self.out_gate = mlp(hidden_dim, hidden_dim, 1)

    def forward(self, x_t, edge_index, edge_attr, noise_level, node_scalars, global_cond=None):
        # x_t: (N, 2) noisy positions (CoM-free)
        # node_scalars: (N, n_scalar_feats) — particle size, type (as 0/1), any other features
        N = x_t.shape[0]
        t_emb = self.time_embed(noise_level.expand(N, 1) if noise_level.shape[0] != N else noise_level)
        # t_emb: (N, hidden_dim)

        scalar_emb = self.scalar_proj(node_scalars.float())  # (N, hidden_dim)

        feats = [t_emb, scalar_emb]
        if self.global_cond_dim > 0:
            assert global_cond is not None
            feats.append(global_cond)

        h = self.input_mlp(torch.cat(feats, dim=-1))
        #This above transform is what takes h to (N, hidden_dim) so that it is compatible with the EGNNLayer class
        x = x_t.clone()
        x_in = x_t
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        gate = self.out_gate(h)
        pred_noise = gate * (x - x_in)
        return pred_noise