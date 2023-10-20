import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from config import get_gpt_config,get_bert_config,get_vit_config
config_gpt = get_gpt_config()
config_vit = get_vit_config()

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(out_dim, out_dim * 3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, x, y=None, mask=None):
        y = x if y is None else y

        qkv = self.linear(x)
        q = qkv[:, :, :self.out_dim]
        k = qkv[:, :, self.out_dim:self.out_dim * 2]
        v = qkv[:, :, self.out_dim * 2:]

        q, k, v = [self.split_heads(t) for t in (q, k, v)]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        scores = self.attention(q, k, v, mask)
        scores = scores.transpose(1, 2).contiguous().view(scores.shape[0], -1, self.out_dim)
        out = self.out(scores)
        return out

    def attention(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1))
        scores /= math.sqrt(q.shape[-1])

        # Mask
        scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = scores.matmul(v)
        return output

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()
        
        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?
        
    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings 
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output
    
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()
        
        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
        )
    
    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
# 1. Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        # 4. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    # 5. Create a forward() method  
    def forward(self, x):
        
        # 6. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x 
        
        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x 
        
        return x
    

class MultiHeadAttentionGPT(nn.Module):
    def __init__(self, num_heads, head_size, num_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                self.AttentionHeadGPT(
                    head_size=head_size,
                    num_embed=num_embed,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Output of the self-attention
        out = torch.cat([self.head(x) for self.head in self.heads], dim=-1)
        # Apply the linear projection layer
        out = self.dropout(self.proj(out))
        return out

    class AttentionHeadGPT(nn.Module):
        """
        One head of the self-attention layer
        """

        def __init__(self, head_size, num_embed, block_size, dropout):
            super().__init__()
            self.key = nn.Linear(num_embed, head_size, bias=False)
            self.query = nn.Linear(num_embed, head_size, bias=False)
            self.value = nn.Linear(num_embed, head_size, bias=False)
            # tril is a lower triangular matrix. It is not a parameter
            # of the model, so we assign it to the module using register_buffer
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

            # Let's also add dropout
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            # Compute attention scores
            # (B, T, C) @ (B, C, T) -> (B, T, T)
            wei = q @ k.transpose(-2, -1) * C ** -0.5
            # Tril matrix (lower triangular matrix) is used to mask 
            # future positions (setting them to -inf) so that the
            # decoder "learns" to predict next words
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = self.dropout(wei)
            # Weighted aggregation of the values
            v = self.value(x)
            out = wei @ v  # (B, T, T) @ (B, T, C) ---> (B, T, C)
            return out

class FeedForward1(nn.Module):
    """
    A simple linear layer followed by ReLu
    """

    def __init__(self, num_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # in the Attention is All You Need paper
            # authors are using the size of the ffwd layer 2048
            # and the output of the model is 512
            # so we apply the same factor of 4
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            # apply the linear projection layer
            nn.Linear(4 * num_embed, num_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    This calss will group together MultiHead Attention and
    FeedForward NN, so that we can copy it in Transformer
    """

    def __init__(self, num_heads, block_size, num_embed, dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MultiHeadAttentionGPT(
            num_heads=num_heads,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward1(num_embed=num_embed, dropout=dropout)
        # add the layer normalization
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        # "x +" is the skip (or residual) connection
        # it helps with optimization
        # also we apply layer normalization before self-attention
        # and feed-forward (a reshufle from original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
            
# Positional Embedding
class PositionalEmbedding(nn
                          .Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len


patch_size=16

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]



# 1. Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    # 11. Create a forward() method
    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index
        return x    
        
        
class Transformer(nn.Module):
    def __init__(self, model_type, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, dropout=0.1):
        super().__init__()

        self.model_type = model_type  # 'bert' or 'gpt'

        if self.model_type == 'bert':
            self.embeddings = nn.Embedding(n_embeddings, embed_size)
            self.pe = PositionalEmbedding(embed_size, seq_len)

            encoders = []
            for i in range(n_code):
                encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
            self.encoders = nn.ModuleList(encoders)

            self.norm = nn.LayerNorm(embed_size)
            self.linear = nn.Linear(embed_size, n_embeddings, bias=False)

        elif self.model_type == 'gpt':
            self.vocab_size = n_embeddings
            self.num_embed = embed_size
            self.block_size = seq_len
            self.num_heads = n_heads
            self.num_layers = n_code
            self.dropout = dropout

            self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
            self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
            self.blocks = nn.Sequential(
                *[TransformerBlock(
                    num_heads=self.num_heads,
                    block_size=self.block_size,
                    num_embed=self.num_embed,
                    dropout=self.dropout,
                ) for _ in range(self.num_layers)]
            )

            self.ln_f = nn.LayerNorm(self.num_embed)
            self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

        elif self.model_type == 'vit':
            self.vit_model = ViT()
        

    def forward(self, idx, targets=None):
        if self.model_type == 'bert':
            x = self.embeddings(idx)
            x = x + self.pe(x)
            for encoder in self.encoders:
                x = encoder(x)
            x = self.norm(x)
            x = self.linear(x)
            return x

        elif self.model_type == 'gpt':
            B, T = idx.shape
            # idx and targets are (B,T) tensor of integers
            # the token_emb is (B, T, C), C = NUM_EMBED
            token_emb = self.token_embedding_table(idx)
            # (T, C)
            posit_emb = self.position_embedding_table(torch.arange(T, device=config_gpt['DEVICE']))

            x = token_emb + posit_emb
            # apply one head of self-attention
            x = self.blocks(x)
            # (B, T, vocab_size)
            logits = self.lm_head(x)
            # compute the loss
            if targets != None:
                # cross_entropy accepts inputs in a (batch_size, num_classes)
                # so we need to reformat our logits dimensions to
                # (batch_size * time, dim_vocabulary), time = block_size
                B, T, C = logits.shape
                logits = torch.reshape(logits, (B * T, C))
                targets = torch.reshape(targets, (B * T,))
                loss = F.cross_entropy(logits, targets)
            else:
                loss = None
            return logits, loss

        elif self.model_type == 'vit':
            x = self.vit_model(idx)  # Forward pass through ViT model
            return x
        
    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self.forward(idx_crop)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx