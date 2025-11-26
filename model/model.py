import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    '''
    Classic sine/cosine positional encoding.

        Injects token index information so attention can reason about order.
    '''

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)                             # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # no gradients; moves with .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Add precomputed positions to embeddings (expects [B, T, d_model]).
        '''
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)



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
        NOTE: "How many separate views of attention are combined"

    num_layers : int, optional
        Number of stacked encoder and decoder layers (default: 4).
        NOTE: "Depth of the model"

    ff_dim : int, optional
        Size of the intermediate hidden layer within each Transformer block and projection head (default: 512). 
        NOTE: "How many neurons process each data line"

    dropout : float, optional
        Dropout probability applied to projection layers for regularization (default: 0.1).

    label_smoothing : float, optional
        Label smoothing factor for cross-entropy loss (default: 0.1).
    '''

    def __init__(
            self,
            vocab_size: int,
            pw_vocab_size: int,
            pad_id: int,
            sos_id: int,
            eos_id: int,
            d_model: int = 256,
            n_heads: int = 8,
            num_layers: int = 4,
            ff_dim: int = 512,
            dropout: float = 0.1,
            label_smoothing: float = 0.1
    ) -> None:
        super().__init__()

        # anti-collapse hyperparameter
        self.label_smoothing = label_smoothing

        # ---- embedding layers ----
        # convert integer-based tokens into dense vector embeddings that can carry meaning
        # padding_idx ensures that padded positions are ignored during training updates
        self.hash_embed = nn.Embedding(vocab_size, d_model)
        self.pw_embed = nn.Embedding(pw_vocab_size, d_model, padding_idx = pad_id)
        self.hash_pos_enc = PositionalEncoding(d_model, max_len = 64)   # plenty for 16-byte hashes
        self.pw_pos_enc = PositionalEncoding(d_model, max_len = 256)    # passwords rarely exceed this

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
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # ---- deep nonlinear output projection ----
        # transforms decoder outputs through multiple layers before producing logits.
        # this prevents the model from relying on a shallow linear mapping.
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(ff_dim),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(ff_dim // 2),
            nn.Linear(ff_dim // 2, pw_vocab_size)
        )

        # store the padding ID for later masking
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        # cache causal masks so we don't reallocate every forward pass
        self._cached_causal_mask: torch.Tensor | None = None
        self._cached_mask_size = 0
        self._cached_mask_device: torch.device | None = None

        # initialize weights for gradient stability
        self._init_weights()
    

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

        # embed raw integer tokens into dense vectors
        hash_emb = self.hash_embed(hash_batch)      # [B, 16, d_model]
        hash_emb = self.hash_pos_enc(hash_emb)      # inject byte positions
        pw_emb = self.pw_embed(decoded_pw_batch)    # [B, T-1, d_model]
        pw_emb = self.pw_pos_enc(pw_emb)            # inject character positions

        # build padding masks
        pw_pad_mask = (decoded_pw_batch == self.pad_id)      # [B, T-1]

        # causal mask for decoder reused from cache
        n = decoded_pw_batch.size(1)
        causal_mask = self._get_causal_mask(n, decoded_pw_batch.device)  # [T-1, T-1]

        # Encode hash (NO MASK NEEDED)
        hash_encoded = self.encoder(hash_emb)  # [B, 16, d_model]
        hash_encoded = self.encoder_projection(hash_encoded)

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
        Compute cross-entropy loss with label smoothing.

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
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        # cross-entropy loss with label smoothing
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index = self.pad_id,
            label_smoothing = self.label_smoothing
        )

        return loss

    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        '''
        Return an upper-triangular causal mask of the requested size, reusing cached storage when possible.
        '''
        need_new_mask = (
            self._cached_causal_mask is None
            or self._cached_mask_size < size
            or self._cached_mask_device != device
        )
        if need_new_mask:
            mask = torch.triu(torch.ones(size, size, device = device), diagonal = 1).bool()
            self._cached_causal_mask = mask
            self._cached_mask_size = size
            self._cached_mask_device = device
        return self._cached_causal_mask[:size, :size]


    @torch.no_grad()
    def generate(self, hash_batch: torch.Tensor, max_length: int = 32, temperature: float = 1.0, repetition_penalty: float = 1.0) -> torch.Tensor:
        '''
        Autoregressively generate passwords from hash inputs (inference mode).

        Parameters:
        -----------
        hash_batch : torch.Tensor
            input hash tokens, shape [B, 16]

        max_length : int, optional
            maximum number of tokens to generate (default: 32)

        temperature : float, optional
            sampling temperature for controlling randomness (default: 1.0)

        repetition_penalty : float, optional
            penalty applied to tokens that were already generated (default: 0.0 = disabled)

        Returns:
        --------
        torch.Tensor
            generated password token IDs, shape [B, T] where T <= max_length
        '''

        self.eval()  # ensure model is in eval mode
        B = hash_batch.size(0)
        device = hash_batch.device

        # encode hash once
        hash_emb = self.hash_embed(hash_batch)           # [B, 16, d_model]
        hash_emb = self.hash_pos_enc(hash_emb)           # add byte positions so encoder sees order
        hash_encoded = self.encoder(hash_emb)            # [B, 16, d_model]
        hash_encoded = self.encoder_projection(hash_encoded)

        # initialize generated sequence with <SOS> token
        generated = torch.full((B, 1), self.sos_id, dtype = torch.long, device = device)  # [B, 1]

        # generate tokens one at a time
        for _ in range(max_length - 1):  # -1 because we already have <SOS>
            # embed current sequence
            pw_emb = self.pw_embed(generated)  # [B, current_len, d_model]
            pw_emb = self.pw_pos_enc(pw_emb)

            # create causal mask for current sequence length (cached)
            current_len = generated.size(1)
            causal_mask = self._get_causal_mask(current_len, device)

            # decode with current sequence
            pw_decoded = self.decoder(
                tgt = pw_emb,
                memory = hash_encoded,
                tgt_mask = causal_mask
            )  # [B, current_len, d_model]

            # get logits for next token
            # (only need last position)
            next_token_logits = self.output_head(pw_decoded[:, -1, :])  # [B, pw_vocab_size]

            # apply repetition penalty if enabled
            if repetition_penalty >= 1.0:
                # penalize tokens that were already generated
                # (divide logits by penalty factor to reduce their probability)
                for i in range(B):
                    # get unique tokens already generated in this sequence (excluding <SOS>)
                    already_generated = generated[i, 1:].unique()  # skip <SOS> token
                    # reduce logits for already-generated tokens
                    next_token_logits[i, already_generated] = next_token_logits[i, already_generated] / repetition_penalty

            # apply temperature and sample/select next token
            if temperature == 1.0:
                # greedy decoding (deterministic)
                next_token = next_token_logits.argmax(dim = -1, keepdim = True)  # [B, 1]
            else:
                # temperature-scaled sampling
                next_token_probs = F.softmax(next_token_logits / temperature, dim = -1)  # [B, pw_vocab_size]
                next_token = torch.multinomial(next_token_probs, num_samples=1)  # [B, 1]

            # append next token to sequence
            generated = torch.cat([generated, next_token], dim = 1)  # [B, current_len + 1]

            # check if all sequences have generated <EOS>
            if (next_token == self.eos_id).all():
                break

        return generated  # [B, T] where T <= max_length

    def _init_weights(self) -> None:
        '''
        Initialize model weights for gradient stability.

        Uses Xavier/Glorot initialization for linear layers and small constant
        initialization for embeddings to prevent gradient explosion.
        '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Small uniform initialization for embeddings
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    # Ensure padding embeddings stay zero
                    module.weight.data[module.padding_idx].zero_()

