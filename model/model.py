import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PAD_ID, SOS_ID, EOS_ID, ALLOWED_PW_CHARS


class OptimusPrime(nn.Module):
    '''
    Transformer encoder–decoder model that learns to map a hash digest to corresponding plaintext password.

        This extended architecture adds nonlinear feedforward projection heads on both
        the encoder output and the final decoder output. These additional layers allow
        the model to learn richer, high-dimensional relationships between hash byte
        patterns and character-level password distributions—going beyond simple linear
        associations to approximate deeper statistical correlations.

    Parameters:
    -----------
    vocab_size : int
        Number of unique hash byte tokens (typically 257 = 256 values + 1 padding).

    pw_vocab_size : int
        Number of possible password characters in the dataset.

    pad_id : int
        Padding token index used to mask padded values during self-attention and loss.

    d_model : int, optional
        Dimensionality of token embeddings and hidden representations (default: 256).
        NOTE: "Width of each data line"

    n_heads : int, optional
        Number of attention heads per Transformer layer (default: 8).
        NOTE: “How many separate views of attention are combined”

    num_layers : int, optional
        Number of stacked encoder and decoder layers (default: 4).
        NOTE: “Depth of the model”

    ff_dim : int, optional
        Size of the intermediate hidden layer within each Transformer block and projection head (default: 512). 
        Increasing this expands model capacity.
        NOTE: “How many neurons process each data line”

    dropout : float, optional
        Dropout probability applied to projection layers for regularization (default: 0.1).

    Notes:
    ------
    The model operates in five conceptual stages:
        1. Hash bytes and password tokens are embedded into dense vector spaces.
        2. Hash embeddings are processed through a Transformer encoder to learn
           latent structural patterns in the digest.
        3. The encoded hash representation passes through a nonlinear projection MLP,
           enriching it with higher-order statistical features.
        4. The Transformer decoder generates password token representations conditioned
           on these encoded features.
        5. A deep multi-layer projection head transforms decoder outputs into logits
           over the password vocabulary, enabling categorical prediction via
           cross-entropy loss.

    This architecture is suitable for tasks that require learning nonlinear,
    statistically grounded mappings between two symbolic domains, such as
    cryptographic inversion experiments or generative password modeling.
    '''

    def __init__(
            self, 
            vocab_size: int = 257, 
            pw_vocab_size: int = len(ALLOWED_PW_CHARS), 
            pad_id: int = PAD_ID, 
            sos_id: int = SOS_ID,
            eos_id: int = EOS_ID,
            d_model: int = 256, 
            n_heads: int = 8, 
            num_layers: int = 4, 
            ff_dim: int = 512, 
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        # ---- embedding layers ----
        # convert integer-based tokens into dense vector embeddings that can carry meaning
        # padding_idx ensures that padded positions are ignored during training updates
        self.hash_embed = nn.Embedding(vocab_size, d_model)
        self.pw_embed = nn.Embedding(pw_vocab_size, d_model, padding_idx = pad_id)

        # ---- transformer encoder ----
        # learns to model statistical relationships among hash bytes
        # each encoder layer contains multi-head attention + feedforward sublayers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = n_heads, 
            dim_feedforward = ff_dim, 
            dropout = dropout, 
            batch_first = True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        # ---- transformer decoder ----
        # receives the enriched hash representation and predicts password token embeddings
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model, 
            nhead = n_heads,
            dim_feedforward = ff_dim,
            dropout = dropout, 
            batch_first = True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)

        # ---- nonlinear encoder projection ----
        # an MLP applied after the encoder to expand representational power
        # this lets the model learn higher-order (nonlinear) statistical dependencies
        self.encoder_projection = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

        # ---- deep nonlinear output projection ----
        # transforms decoder outputs through multiple layers before producing logits.
        # this prevents the model from relying on a shallow linear mapping.
        self.output_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, pw_vocab_size)
        )

        # store the padding ID for later masking
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
    

    def forward(self, hash_batch: torch.Tensor, pw_batch: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the model.

            encode the hash bytes
            decode conditioned on that representation
            output logits for each possible password token

        Parameters:
        -----------
        hash_batch : torch.LongTensor
            Input hash tokens, shape [B, 16]

        pw_batch : torch.LongTensor
            Password tokens INCLUDING <SOS> and <EOS>, shape [B, T]

        Returns:
        --------
        torch.Tensor
            Logits of shape [B, T, pw_vocab_size]
                (unnormalized model predictions)
        '''

        # Decoder input: everything EXCEPT last token (<EOS>)
        # [<SOS>, 'h', 'e', 'l', 'l', 'o', <EOS>] -> [<SOS>, 'h', 'e', 'l', 'l', 'o']
        decoded_pw_batch = pw_batch[:, :-1]  # [B, T-1]

        # Targets: everything EXCEPT first token (<SOS>)
        # [<SOS>, 'h', 'e', 'l', 'l', 'o', <EOS>] → ['h', 'e', 'l', 'l', 'o', <EOS>]
        # (used in compute_loss)

        # embed raw integer tokens into dense vectors
        hash_emb = self.hash_embed(hash_batch)      # [B, 16, d_model]
        pw_emb = self.pw_embed(decoded_pw_batch)    # [B, T-1, d_model]

        # build padding masks
        pw_pad_mask = (decoded_pw_batch == self.pad_id)      # [B, T-1]
        
        # causal mask for decoder
        n = decoded_pw_batch.size(1)
        causal_mask = torch.triu(torch.ones(n, n), diagonal = 1).bool().to(decoded_pw_batch.device)  # [T-1, T-1]

        # Encode hash (NO MASK NEEDED)
        hash_encoded = self.encoder(hash_emb)  # [B, 16, d_model]
        # NOTE: src_key_padding_mask defaults to None

        # decode password
        pw_decoded = self.decoder(
            tgt = pw_emb,
            memory = hash_encoded,
            tgt_mask = causal_mask,
            tgt_key_padding_mask = pw_pad_mask,
        )  # [B, T-1, d_model]

        # project each decoder output to logits over password vocabulary
        logits = self.output_head(pw_decoded)   # [B, T-1, pw_vocab_size]
        return logits                           # logit = raw / unnormalized output of model before converted to probability


    def compute_loss(self, logits: torch.Tensor, pw_batch: torch.Tensor) -> torch.Tensor:
        '''
        Compute cross-entropy loss while ignoring padded positions.

        Parameters:
        -----------
        logits : torch.Tensor
            Model output logits, shape [B, T-1, pw_vocab_size]

        pw_batch : torch.Tensor
            Full password tokens INCLUDING <SOS> and <EOS>, shape [B, T]

        Returns:
        --------
        torch.Tensor
            Scalar loss averaged over non-padded tokens
        '''

        # targets: skip <SOS> (first token)
        targets = pw_batch[:, 1:]  # [B, T-1]
    
        B, T, V = logits.shape  # batch size, sequence length, vocabulary size
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)

        # ignore_index masks <PAD>, but <EOS> is now learned
        return F.cross_entropy(logits, targets, ignore_index = self.pad_id)
