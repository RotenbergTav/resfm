import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential, Module, Identity
from utils.sparse_utils import SparseMat
from utils.pos_enc_utils import get_embedder


def get_linear_layers(feats, final_layer=False, batchnorm=True):
    layers = []

    # Add layers
    for i in range(len(feats) - 2):
        layers.append(Linear(feats[i], feats[i + 1]))

        if batchnorm:
            layers.append(BatchNorm1d(feats[i + 1], track_running_stats=False))

        layers.append(ReLU())

    # Add final layer
    layers.append(Linear(feats[-2], feats[-1]))
    if not final_layer:
        if batchnorm:
            layers.append(BatchNorm1d(feats[-1], track_running_stats=False))

        layers.append(ReLU())

    return Sequential(*layers)


class Parameter3DPts(torch.nn.Module):
    def __init__(self, n_pts):
        super().__init__()

        # Init points randomly
        pts_3d = torch.normal(mean=0, std=0.1, size=(3, n_pts), requires_grad=True)

        self.pts_3d = torch.nn.Parameter(pts_3d)

    def forward(self):
        return self.pts_3d


class SetOfSetLayer(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)
        self.lin_n = Linear(d_in, d_out)
        self.lin_m = Linear(d_in, d_out)
        self.lin_both = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        out_all = self.lin_all(x.values)  # [all_points_everywhere, d_in] -> [all_points_everywhere, d_out]

        mean_rows = x.mean(dim=0) # [m,n,d_in] -> [n,d_in]
        out_rows = self.lin_n(mean_rows)  # [n,d_in] -> [n,d_out]  # each track's mean representation gets weighted

        mean_cols = x.mean(dim=1) # [m,n,d_in] -> [m,d_in]
        out_cols = self.lin_m(mean_cols)  # [m,d_in] -> [m,d_out]  # each camera's mean representation gets weighted

        out_both = self.lin_both(x.values.mean(dim=0, keepdim=True))  # [1,d_in] -> [1,d_out]

        new_features = (out_all + out_rows[x.indices[1], :] + out_cols[x.indices[0], :] + out_both) / 4  # [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class SetOfSetAttentionLayer(Module):
    """
    Multi-head attention over row/col/global summaries (perm-equivariant).
    Extra params (from conf.model.layer_extra_params):
      heads (int) = 4
      attn_dropout (float) = 0.1
      temperature (float) = 1.0    # <1 = sharper softmax, >1 = softer
      use_global (bool) = True
    """
    def __init__(self, d_in, d_out, **kwargs):
        super(SetOfSetAttentionLayer, self).__init__()

        # ---- parse extra params ----
        self.h = int(kwargs.get("heads", 4))
        self.tau = float(kwargs.get("temperature", 1.0))
        self.use_global = bool(kwargs.get("use_global", True))
        self.attn_drop = torch.nn.Dropout(float(kwargs.get("attn_dropout", 0.1)))


        if d_out % self.h != 0:
            raise ValueError(f"d_out={d_out} must be divisible by heads={self.h}")
        self.d_head = d_out // self.h

        # Local residual term
        self.lin_all = Linear(d_in, d_out)

        # Q/K/V projections
        self.q_proj = Linear(d_in, d_out)
        self.k_row = Linear(d_in, d_out); self.v_row = Linear(d_in, d_out)
        self.k_col = Linear(d_in, d_out); self.v_col = Linear(d_in, d_out)
        self.k_glb = Linear(d_in, d_out); self.v_glb = Linear(d_in, d_out)

    def _split_heads(self, x):
        # x: [N, d_out] -> [N, H, d_head]
        return x.view(x.shape[0], self.h, self.d_head)

    def forward(self, x: SparseMat):
        # Summaries (reuse SparseMat reductions, preserving equivariance)
        row_ctx = x.mean(dim=0)                       # [n, d_in]  per-track
        col_ctx = x.mean(dim=1)                       # [m, d_in]  per-camera
        glb_ctx = x.values.mean(dim=0, keepdim=True)  # [1, d_in]  global

        # Projections
        Q  = self.q_proj(x.values)                    # [nnz, d_out]
        Kr = self.k_row(row_ctx); Vr = self.v_row(row_ctx)
        Kc = self.k_col(col_ctx); Vc = self.v_col(col_ctx)
        Kg = self.k_glb(glb_ctx); Vg = self.v_glb(glb_ctx)

        # Gather per observation (i,j)
        j_idx = x.indices[1]; i_idx = x.indices[0]
        Kr_sel = Kr[j_idx]; Vr_sel = Vr[j_idx]        # [nnz, d_out]
        Kc_sel = Kc[i_idx]; Vc_sel = Vc[i_idx]
        Kg_sel = Kg.expand_as(Q); Vg_sel = Vg.expand_as(Q)

        # Split into heads
        Qh  = self._split_heads(Q)
        Krh = self._split_heads(Kr_sel);  Vrh = self._split_heads(Vr_sel)
        Kch = self._split_heads(Kc_sel);  Vch = self._split_heads(Vc_sel)
        Kgh = self._split_heads(Kg_sel);  Vgh = self._split_heads(Vg_sel)

        # Scaled dot-product scores per head
        scale = self.d_head ** 0.5
        s_row = (Qh * Krh).sum(-1) / scale            # [nnz, H]
        s_col = (Qh * Kch).sum(-1) / scale            # [nnz, H]

        if self.use_global:
            s_glb = (Qh * Kgh).sum(-1) / scale        # [nnz, H]
            logits = torch.stack([s_row, s_col, s_glb], dim=-1) / self.tau  # [nnz,H,3]
        else:
            logits = torch.stack([s_row, s_col], dim=-1) / self.tau         # [nnz,H,2]

        w = torch.softmax(logits, dim=-1)
        w = self.attn_drop(w)

        # Weighted sum (per head), then merge heads
        if self.use_global:
            ctx = (w[..., 0:1] * Vrh) + (w[..., 1:2] * Vch) + (w[..., 2:3] * Vgh)
        else:
            ctx = (w[..., 0:1] * Vrh) + (w[..., 1:2] * Vch)
        ctx = ctx.reshape(ctx.shape[0], -1)           # [nnz, d_out]

        # Residual + return SparseMat
        out_all = self.lin_all(x.values)
        new_features = out_all + ctx                  # [nnz, d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)

class ProjLayer(Module):
    def __init__(self, d_in, d_out):
        super(ProjLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class NormalizationLayer(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        # norm_features = norm_features / norm_features.std(dim=0, keepdim=True)
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class ActivationLayer(Module):
    def __init__(self):
        super(ActivationLayer, self).__init__()
        self.relu = ReLU()

    def forward(self, x):
        new_features = self.relu(x.values)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class IdentityLayer(Module):
    def forward(self, x):
        return x


class EmbeddingLayer(Module):
    """
    Applies Fourier PE to the first 'uv_dim' channels (default 2),
    concatenates the remaining channels unchanged, and optionally projects
    the visual tail to control capacity.
    """
    def __init__(self, multires: int, in_dim_total: int, uv_dim: int = 2, vis_proj_dim: int = 0):
        super(EmbeddingLayer, self).__init__()
        self.uv_dim = uv_dim
        self.tail_dim = max(in_dim_total - uv_dim, 0)
        if multires > 0:
            self.embed, self.uv_emb_dim = get_embedder(multires, uv_dim)
        else:
            self.embed, self.uv_emb_dim = (Identity(), uv_dim)
        self.vis_proj = None
        self.out_tail_dim = self.tail_dim
        if self.tail_dim > 0 and vis_proj_dim and vis_proj_dim > 0 and vis_proj_dim != self.tail_dim:
            self.vis_proj = Linear(self.tail_dim, vis_proj_dim)
            self.out_tail_dim = vis_proj_dim
        self.d_out = self.uv_emb_dim + self.out_tail_dim

    def forward(self, x):
        vals = x.values
        uv = vals[:, :self.uv_dim]
        rest = vals[:, self.uv_dim:] if self.tail_dim > 0 else None
        uv_emb = self.embed(uv)
        if rest is not None and self.vis_proj is not None:
            rest = self.vis_proj(rest)
        new_features = uv_emb if rest is None else torch.cat([uv_emb, rest], dim=-1)
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)
