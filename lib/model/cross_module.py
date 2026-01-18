import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#* from nerf
class PosEmbedder:
    def __init__(self, input_dims=3, multires=10):
        self.kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
#* from transformer
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CrossModule(nn.Module):
    def __init__(self, in_HW, hid_dim=256, num_layers=1):
        super(CrossModule, self).__init__()
        self.num_force = 32
        proj_dim = int(hid_dim / (in_HW**2 / self.num_force))

        self.proj_hand = nn.Conv2d(256, proj_dim, 3, 1, 1)
        self.proj_obj = nn.Conv2d(256, proj_dim, 3, 1, 1)

        self.gravity_embedder = PosEmbedder(input_dims=3, multires=10)
        self.gravity_proj = nn.Linear(self.gravity_embedder.out_dim, hid_dim)
        self.pose_embedder = PositionalEncoding(hid_dim)

        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hid_dim, nhead=2),
            num_layers=num_layers,
        )

        self.init()
    
    def init(self,):
        nn.init.kaiming_normal_(self.proj_hand.weight)
        nn.init.constant_(self.proj_hand.bias, 0)
        nn.init.kaiming_normal_(self.proj_obj.weight)
        nn.init.constant_(self.proj_obj.bias, 0)
        nn.init.kaiming_normal_(self.gravity_proj.weight)
        nn.init.constant_(self.gravity_proj.bias, 0)

        
    def forward(self, x_hand, x_obj, gravity):
        """ x_hand: (bs, 256, h, w)
            x_obj: (bs, 256, h, w)
        """
        bs, c, h, w = x_hand.shape
        x_hand = self.proj_hand(x_hand).view(bs, self.num_force, -1)
        x_obj = self.proj_obj(x_obj).view(bs, self.num_force, -1)
        # x_obj = F.adaptive_avg_pool1d(x_obj.transpose(1,2), 1).transpose(1,2)
        
        enc_gravity = self.gravity_embedder.embed(gravity)
        enc_gravity = self.gravity_proj(enc_gravity)

        x = torch.cat([x_hand, x_obj, enc_gravity], dim=1)
        x = self.pose_embedder(x)
        x = self.attn(x)

        y_hand, y_obj, y_gravity = torch.split(x, [self.num_force, self.num_force, 1], dim=1)
        return y_hand, y_obj, y_gravity
