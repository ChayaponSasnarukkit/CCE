import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import drop_path, trunc_normal_

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Attention_Temporal_Centered(nn.Module):
    """
    Bidirectional Hierarchical Temporal Attention (HTA).
    Focuses the $T/4$ and $T/2$ hierarchy strictly on the middle of the window.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.qkv_4 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_8 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_16 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.proj_4 = nn.Linear(dim, dim)
        self.proj_8 = nn.Linear(dim, dim)
        self.proj_16 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, T, C = x.shape
        
        # Center-anchored sequence lengths
        l_4 = T // 4
        l_8 = T // 2
        mid = T // 2
        
        # Calculate start and end indices to perfectly center the slices
        start_4 = mid - (l_4 // 2)
        end_4 = start_4 + l_4
        start_8 = mid - (l_8 // 2)
        end_8 = start_8 + l_8
        
        # Extract symmetric slices from the middle of the sequence
        x_4 = x[:, start_4:end_4, :]
        x_8 = x[:, start_8:end_8, :]
        x_16 = x
        
        def process_qkv(qkv_linear, x_input):
            qkv = qkv_linear(x_input)
            qkv = rearrange(qkv, "b t (qkv num_heads c) -> qkv b num_heads t c", qkv=3, num_heads=self.num_heads)
            return qkv[0], qkv[1], qkv[2]
            
        q_4, k_4, v_4 = process_qkv(self.qkv_4, x_4)
        q_8, k_8, v_8 = process_qkv(self.qkv_8, x_8)
        q_16, k_16, v_16 = process_qkv(self.qkv_16, x_16)
        
        def compute_attn(q, k, v):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            out = attn @ v
            return rearrange(out, "b num_heads t c -> b t (num_heads c)")

        # Compute attention for each hierarchy level
        x_4_out = compute_attn(q_4, k_4, v_4)
        x_8_out = compute_attn(q_8, k_8, v_8)
        x_16_out = compute_attn(q_16, k_16, v_16)

        x_4_proj = self.proj_4(x_4_out)
        
        # Inject x_4 into the precise middle of x_8
        offset_4_in_8 = (l_8 - l_4) // 2
        x_8_out_mod = x_8_out.clone() # Clone to avoid inplace autograd errors
        x_8_out_mod[:, offset_4_in_8 : offset_4_in_8 + l_4, :] = (
            0.5 * x_8_out[:, offset_4_in_8 : offset_4_in_8 + l_4, :] + 0.5 * x_4_proj
        )
        x_8_proj = self.proj_8(x_8_out_mod)
        
        # Inject x_8 into the precise middle of x_16 (full sequence)
        offset_8_in_16 = (T - l_8) // 2
        x_16_out_mod = x_16_out.clone()
        x_16_out_mod[:, offset_8_in_16 : offset_8_in_16 + l_8, :] = (
            0.5 * x_16_out[:, offset_8_in_16 : offset_8_in_16 + l_8, :] + 0.5 * x_8_proj
        )
        
        return self.proj_drop(self.proj_16(x_16_out_mod))

class TemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.temporal_norm = norm_layer(dim)
        self.temporal_attn = Attention_Temporal_Centered(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.temporal_attn(self.temporal_norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TemporalWindowClassifier(nn.Module):
    def __init__(self, window_size=32, embed_dim=768, num_classes=2, depth=4, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        assert window_size % 4 == 0, "Window size must be cleanly divisible by 4 for HTA math."
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Temporal Positional Embedding
        self.time_embed = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TemporalBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=True, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            ) for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        trunc_normal_(self.time_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x input shape: (B, T, C) -> e.g. (Batch, 32 frames, 768 dim)
        B, T, C = x.shape
        assert T == self.window_size, f"Input sequence length {T} must match model window_size {self.window_size}"
        
        x = x + self.time_embed
        x = self.time_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x) # Shape remains (B, T, C)
        
        # Extract the highly enriched middle frame
        mid_idx = T // 2
        mid_feature = x[:, mid_idx, :] # Shape: (B, C)
        
        # Classify directly using the middle frame
        return self.head(mid_feature)