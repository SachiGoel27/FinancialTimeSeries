import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MLP(nn.Module):
    def __init__(self, emb_type, model_in_size, d_model, e_layers):
        super(MLP, self).__init__()
        self.emb_type = emb_type
        self.model_in_size = model_in_size
        self.d_model = d_model
        self.e_layers = e_layers

        self.mlp = nn.Sequential(
            nn.Linear(self.model_in_size, self.d_model), GELUComplex() if self.emb_type=='freq' else nn.GELU(),
            *[nn.Sequential(nn.Linear(self.d_model, self.d_model), GELUComplex() if self.emb_type=='freq' else nn.GELU()) \
              for _ in range(self.e_layers+2)],
            nn.Linear(self.d_model, self.model_in_size))
        
    def forward(self, x):
        return self.mlp(x)
    

class RNN(nn.Module):
    def __init__(self, emb_type, model_in_size, d_model, e_layers, dropout):
        super(RNN, self).__init__()
        self.emb_type = emb_type
        self.model_in_size = model_in_size
        self.d_model = d_model
        self.e_layers = e_layers
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.model_in_size, hidden_size=self.d_model, num_layers=self.e_layers, \
                          batch_first=True, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(in_features=2*self.d_model, out_features=self.model_in_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x
    

class Transformer(nn.Module):
    def __init__(self, emb_type, model_in_size, d_ff, e_layers, n_heads, activation, dropout=0.1, factor=5):
        super(Transformer, self).__init__()
        self.emb_type = emb_type
        self.model_in_size = model_in_size
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.factor = factor
        self.activation = activation


        self.trans = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False), self.model_in_size, self.n_heads),
            self.model_in_size, self.d_ff, dropout=self.dropout, activation=self.activation) for l in range(self.e_layers)], 
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.model_in_size), Transpose(1,2)) if self.emb_type=='patch' else torch.nn.LayerNorm(self.model_in_size))

    def forward(self, x):
        x = self.trans(x)
        return x


# Modules for Transformer
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale*scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x
    

class GELUComplex(nn.Module):
    def __init__(self):
        super(GELUComplex, self).__init__()
        self.gelu_real = nn.GELU()
        self.gelu_img = nn.GELU()

    def forward(self, x):
        gelu_real = self.gelu_real(x.real)
        gelu_imag = self.gelu_img(x.imag)
        return torch.complex(gelu_real, gelu_imag)


class MAGN(nn.Module):
    """
    Modality-Aware Gated Network (MAGN)
    
    Processes multiple data modalities (price, macro, news, etc.) independently,
    applies feature-level gating, and fuses them using a target-conditioned
    gating mechanism. This enables dynamic weighting of modality importance.
    
    Args:
        modalities: dict mapping modality names to input dimensions
        d_model: embedding dimension
        d_ff: hidden dimension for MLPs
        model_in_size: output dimension (same as input for seq2seq)
        use_feature_gating: whether to apply feature-level gates
        use_target_conditioning: whether to condition fusion on target
        shared_mlp: whether to share MLP weights across modalities
        dropout: dropout rate
        activation: activation function ("relu" or "gelu")
    """
    
    def __init__(self, modalities, d_model, d_ff, model_in_size, 
                 use_feature_gating=True, use_target_conditioning=True,
                 shared_mlp=False, dropout=0.1, activation="gelu"):
        super(MAGN, self).__init__()
        
        self.modalities = modalities if isinstance(modalities, dict) else {f"modality_{i}": d for i, d in enumerate(modalities)}
        self.d_model = d_model
        self.d_ff = d_ff
        self.model_in_size = model_in_size
        self.use_feature_gating = use_feature_gating
        self.use_target_conditioning = use_target_conditioning
        self.shared_mlp = shared_mlp
        self.dropout = dropout
        self.activation = F.relu if activation == "relu" else F.gelu
        self.num_modalities = len(self.modalities)
        
        # Feature gating per modality
        if self.use_feature_gating:
            self.feature_gates = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.Sigmoid()
                ) for name in self.modalities.keys()
            })
        
        # Modality-specific processing MLPs
        if self.shared_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        else:
            self.modality_mlps = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                ) for name in self.modalities.keys()
            })
        
        # Target-conditioned fusion gate
        fusion_input_dim = d_model * (self.num_modalities + 1)  # All modalities + target
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_input_dim, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.num_modalities)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, model_in_size)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, target_embedding=None):
        """
        Forward pass for MAGN. Can handle both dict and tensor inputs.
        
        Args:
            x: dict of {modality_name: tensor} or tensor [B, T, d_model]
               If tensor, creates modality_embeddings dict automatically
            target_embedding: tensor [B, T, d_model] for conditioning (optional)
        
        Returns:
            output: tensor [B, T, model_in_size]
        """
        
        # Convert tensor input to modality dict if needed
        if not isinstance(x, dict):
            # Use same embedding for all modalities (adapter mode)
            modality_embeddings = {name: x for name in self.modalities.keys()}
        else:
            modality_embeddings = x
        
        return self._forward_impl(modality_embeddings, target_embedding)
    
    def _forward_impl(self, modality_embeddings, target_embedding=None):
        """
        Forward pass for MAGN.
        
        Args:
            modality_embeddings: dict of {modality_name: tensor [B, T, d_model]}
            target_embedding: tensor [B, T, d_model] for conditioning (optional)
        
        Returns:
            output: tensor [B, T, model_in_size]
        """
        
        B, T, D = next(iter(modality_embeddings.values())).shape
        device = next(iter(modality_embeddings.values())).device
        
        # Step 1: Feature gating per modality
        gated_embeddings = {}
        for name, emb in modality_embeddings.items():
            if self.use_feature_gating:
                gate = self.feature_gates[name](emb)
                gated_embeddings[name] = gate * emb
            else:
                gated_embeddings[name] = emb
        
        # Step 2: Process each modality through its MLP
        modality_outputs = {}
        for name, gated_emb in gated_embeddings.items():
            if self.shared_mlp:
                modality_outputs[name] = self.mlp(gated_emb)
            else:
                modality_outputs[name] = self.modality_mlps[name](gated_emb)
        
        # Step 3: Target-conditioned fusion
        modality_list = [modality_outputs[name] for name in self.modalities.keys()]
        
        if self.use_target_conditioning and target_embedding is not None:
            # Concatenate all modalities with target for fusion gate
            fusion_input = torch.cat([target_embedding] + modality_list, dim=-1)
        else:
            # Use mean of modalities as conditioning if target not provided
            if target_embedding is None:
                target_embedding = torch.mean(torch.stack(modality_list, dim=0), dim=0)
            fusion_input = torch.cat([target_embedding] + modality_list, dim=-1)
        
        # Compute fusion weights
        fusion_logits = self.fusion_gate(fusion_input)  # [B, T, num_modalities]
        fusion_weights = torch.softmax(fusion_logits, dim=-1)  # [B, T, num_modalities]
        
        # Step 4: Weighted fusion of modalities
        stacked_modalities = torch.stack(modality_list, dim=2)  # [B, T, num_modalities, d_model]
        fusion_weights_expanded = fusion_weights.unsqueeze(-1)  # [B, T, num_modalities, 1]
        fused = (stacked_modalities * fusion_weights_expanded).sum(dim=2)  # [B, T, d_model]
        
        # Step 5: Output projection
        output = self.output_proj(self.dropout_layer(fused))
        
        return output



class GLU(nn.Module):
    """
    Gated Linear Unit (GLU).
    Acts as a continuous valve to suppress noisy/irrelevant features.
    """
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # The gating signal (0 to 1)
        sig = self.sigmoid(self.linear1(x))
        # The actual transformation
        x = self.linear2(x)
        # Element-wise multiplication
        return sig * x

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from the Temporal Fusion Transformer paper.
    Ideal for financial data due to adaptive complexity and noise suppression.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Linear projection if input size doesn't match output size for the residual skip
        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.skip_layer = nn.Identity()

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        
        self.gate = GLU(self.output_size)
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x):
        """
        Expects input x of shape: [Batch, Time_Steps, Input_Size]
        """
        # 1. Save the original input for the residual skip connection
        residual = self.skip_layer(x)

        # 2. Primary non-linear transformation
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # 3. Apply the Gated Linear Unit to suppress noise
        x = self.gate(x)

        # 4. Add the residual connection and normalize
        x = self.layer_norm(x + residual)

        return x
    
class NBeatsBlock(nn.Module):
    def __init__(self, model_in_size, d_model):
        super(NBeatsBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(model_in_size, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.backcast_head = nn.Linear(d_model, model_in_size)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast_head(h)
        return backcast


class NBeats(nn.Module):
    """
    N-BEATS as a drop-in ff_type for TimeRecipe.
    Each block explains part of the input via a backcast and subtracts it.
    Output is the final unexplained residual, passed to proj_l as normal.
    """
    def __init__(self, model_in_size, d_model, num_blocks=3):
        super(NBeats, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(model_in_size, d_model)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        x: [B, T, model_in_size]
        Returns: [B, T, model_in_size]
        """
        for block in self.blocks:
            backcast = block(x)
            x = x - backcast
        return x
    
class RegimeSwitchingMLP(nn.Module):
    """
    Regime Switching MLP for financial time series.
    
    Replaces a single MLP with a mixture of regime-specific MLPs.
    A gating network learns to softly assign the current input to regimes,
    and the final output is a weighted sum of each regime MLP's output.
    
    No pretraining or regime labels needed - regimes are learned implicitly
    during end-to-end training.
    
    Args:
        model_in_size: input/output feature dimension
        d_model: hidden dimension for each regime MLP
        num_regimes: number of regime-specific MLPs (default 4)
        dropout: dropout rate
    """
    def __init__(self, model_in_size, d_model, num_regimes=4, dropout=0.1):
        super(RegimeSwitchingMLP, self).__init__()
        self.num_regimes = num_regimes
        self.model_in_size = model_in_size

        # One small MLP per regime
        self.regime_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_in_size, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, model_in_size),
                nn.Dropout(dropout)
            ) for _ in range(num_regimes)
        ])

        # Gating network - takes input and outputs soft regime weights
        self.gate = nn.Sequential(
            nn.Linear(model_in_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_regimes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x: [B, T, model_in_size]
        Returns: [B, T, model_in_size]
        """
        # Compute regime weights: [B, T, num_regimes]
        weights = self.gate(x)

        # Compute each regime MLP output: [B, T, model_in_size]
        regime_outputs = torch.stack(
            [mlp(x) for mlp in self.regime_mlps], dim=2
        )  # [B, T, num_regimes, model_in_size]

        # Weighted sum across regimes
        # weights: [B, T, num_regimes] -> [B, T, num_regimes, 1]
        weights = weights.unsqueeze(-1)
        out = (regime_outputs * weights).sum(dim=2)  # [B, T, model_in_size]

        return out