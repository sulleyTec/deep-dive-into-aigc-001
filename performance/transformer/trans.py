'''
    by raymond
'''

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=12):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        assert self.head_dim*num_heads==embed_dim, 'embed_dim must be divisable by num_heads'

        self.values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys= nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.fc = nn.Linear(self.head_dim*self.num_heads, self.embed_dim, bias=False)

        self.cuda_evnet_timing()

    def cuda_evnet_timing(self):
        # Create CUDA events
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize the GPU
        torch.cuda.synchronize()

    def __calc_efficiency(elapsed_time_ms, m1, m2, result):
        bs,l,m,k = m1.shape()
        bs,l,k,n = m2.shape()

        self.band_width = 912/1024/1024
        self.fp16 = self.fp32 = 34.1 # TFLOPS

        inst_time = 0.
        Macs = bs*l*(m*k*n)

        data_amount = m1.numel()*m1.element_size() + \
                      m2.numel()*m1.element_size() + \
                      result.numel()*m1.element_size()

        estimate_time = (inst_time + data_amount/band_width)*1000

    ### input: (bs, seq_len, num_heads, head_dim)
    def mm_perf(self, bs, Q, K, V, mask):

        ### Q.shape = (bs, num_heads, q_seq_len, head_dim)
        Q = Q.permute(0,2,1,3)

        ### Q.shape = (bs, num_heads, head_dim, k_seq_len)
        K = K.permute(0,2,3,1)

        ### energy.shape = (bs, num_heads, q_seq_len, k_seq_len)
        self.start_event.record()
        energy = torch.matmul(Q,K)
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f'attention mm time: {elapsed_time_ms}')

        if mask is not None:
            ### set a large negative value to those positions corresponding to mask=0
            ### when calculate softmax, these values will get to 0
            energy = energy.masked_fill(mask==0, float('-1e20'))

        ### softmax operates on the 3rd dim which is the last dim, 
        ### after this, every row of energy get normalized by softmax
        ### attention.shape = (bs, num_heads, q_seq_len, k_seq_len)
        attention = torch.softmax(energy/(self.embed_dim ** 0.5), dim=3)

        ### V.shape         = (bs, v_seq_len, num_heads, head_dim)
        ### attention.shape = (bs, num_heads, q_seq_len, k_seq_len=v_seq_len)
        ### k_seq_len == v_seq_len != q_seq_len

        # shape = (bs, num_heads, v_seq_len=k_seq_len, head_dim)
        V = V.permute(0,2,1,3) 

        self.start_event.record()
        # out.shape = (bs, num_heads, q_seq_len, head_dim)
        out = torch.matmul(attention, V) 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f'softmax mm time: {elapsed_time_ms}')

        ### out.shape = Q.shape = (bs, q_seq_len, num_heads, head_dim)
        out = out.permute(0,2,1,3).contiguous() 
        ### out.shape = (bs, q_seq_len, embed_dim)
        out = out.view(bs, out.shape[1], -1)
        ### (bs, q_seq_len, embed_dim) @ (embed_dim, embed_dim)
        ### ---> (bs, q_seq_len, embed_dim) = input Q.shape
        out = self.fc(out)

        return out

    def mm_einsum(self, bs, Q, K, V, mask):
        ### performing: Q*(K.T) 
        ### bs-->n, lq-->q, lk-->k, num_heads-->h, head_dim-->d
        ### Q.shape = (bs, lq, num_heads, head_dim) --> (nqhd)
        ### K.shape = (bs, lk, num_heads, head_dim) --> (nkhd)
        ###     generally lk and lq are not equal, lk and lv are the same
        ###
        ### torch.einsum does something like this:
        ###     step 1: exchange dim 1 and 2 of Q and K --> Q.transpose(1,2); K.transpose(1,2)
        ###             ==> Q.shape = (bs, num_heads, lq, head_dim) --> (nhqd)
        ###             ==> K.shape = (bs, num_heads, lk, head_dim) --> (nhkd)
        ###     step 2: do matrix multiplication only on the last two dimensions -->  
        ###             energy = torch.matmul(Q, K.T)
        ###             ==> energy.shape = (bs, num_heads, lq, lk) --> (nhqk)
        ### NOTE: in case onnx does not supply einsum, you have to accomplish by yourself
        ### 
        ### torch.einsum actually can do any high dimensional multiplications of matrices
        ### energy.shape = (bs, num_heads, lq, lk)
        energy = torch.einsum('nqhd, nkhd->nhqk', [Q, K])

        if mask is not None:
            ### set a large negative value to those positions corresponding to mask=0
            ### when calculate softmax, these values will get to 0
            energy = energy.masked_fill(mask==0, float('-1e20'))

        ### softmax operates on the 3rd dim which is the last dim, 
        ### after this, every row of energy get normalized
        ### attention.shape = (bs, num_heads, lq, lk)
        attention = torch.softmax(energy/(self.embed_dim ** 0.5), dim=3)

        ### V.shape   = (bs, lv, num_heads, head_dim)
        ### out.shape = (bs, lq, num_heads, head_dim)
        out = torch.einsum('nhql,nlhd->nqhd',[attention, V]).contiguous()

        out = out.view(bs, out.shape[1], -1)
        out = self.fc(out)

        return out

    ### input :
    ### V.shape = (bs, seq_len, embed_dim)
    def forward(self, Q, K, V, mask, einsum=True):

        bs, _, embed_dim = Q.shape

        ### generally k_seq_len and v_seq_len are not equal, 
        ### k_seq_len and v_seq_len are the same
        q_seq_len = Q.shape[1]
        k_seq_len = K.shape[1]
        v_seq_len = V.shape[1]

        ### QKV MatrixMyl: 
        Q = Q.view(-1, self.embed_dim)
        K = K.view(-1, self.embed_dim)
        V = V.view(-1, self.embed_dim)

        ### (bs*seq_len, embed_dim) @ (embed_dim, embed_dim) 
        ### --> (bs*seq_len, embed_dim)
        Q = self.queries(Q)
        K = self.keys(K)
        V = self.values(V)

        ### split Q,K,V into multi-heads
        Q = Q.reshape(bs, q_seq_len, self.num_heads, self.head_dim)
        K = K.reshape(bs, k_seq_len, self.num_heads, self.head_dim)
        V = V.reshape(bs, v_seq_len, self.num_heads, self.head_dim)

        if einsum:
            return self.mm_einsum(bs, Q, K, V, mask)

        return self.mm_perf(bs, Q, K, V, mask)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super().__init__()

        self.attention = Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        '''
        self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, forward_expansion*embed_dim),
                nn.ReLU(),
                nn.Linear(forward_expansion*embed_dim, embed_dim),
                )
        '''

        self.ff0 = nn.Linear(embed_dim, forward_expansion*embed_dim)
        self.relu = nn.ReLU()
        self.ff1 = nn.Linear(forward_expansion*embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask, einsum=True):

        attention = self.attention(Q,K,V,mask, einsum)
        x = self.dropout(self.norm1(attention+Q))
        #forward = self.feed_forward(x)
        y = self.ff0(x)
        y = self.relu(y)
        y = self.ff1(y)
        out = self.dropout(self.norm2(x+y))

        return out

class Encoder(nn.Module):
    def __init__(self, 
                src_vocab_size, 
                embed_dim, 
                num_layers, 
                num_heads,
                device, 
                forward_expansion,
                dropout,
                max_length):

        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList(
                    [
                        TransformerBlock(
                            embed_dim,
                            num_heads,
                            dropout=dropout,
                            forward_expansion=forward_expansion
                            )
                    for _ in range(num_layers)]
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, einsum=True):
        bs, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(bs, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x)+self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(Q=out, K=out, V=out, mask=mask, einsum=einsum)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout, device):
        super().__init__()

        self.attention = Attention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

        self.transformer_block = TransformerBlock(
            embed_dim, num_heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, K, V, src_mask, trg_mask, einsum=True):

        attention = self.attention(Q=x, K=x, V=x, mask=trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(Q=query, K=K, V=V, mask=src_mask, einsum=einsum)
        return out

class Decoder(nn.Module):
    def __init__(self,
            trg_vocab_size,
            embed_dim,
            num_layers,
            num_heads,
            forward_expansion,
            dropout,
            device,
            max_length):

        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim, 
                    num_heads,
                    forward_expansion,
                    dropout,
                    device
                    )
            for _ in range(num_layers)]
        )

        self.fc = nn.Linear(embed_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask, einsum=True):
        bs, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(bs, seq_len).to(self.device)
        x = self.dropout((self.word_embedding(x)+self.positional_embedding(positions)))

        for layer in self.layers:
            #def forward(self, x, K, V, src_mask, trg_mask):
            x = layer(x, K=enc_out, V=enc_out, 
                    src_mask=src_mask, trg_mask=trg_mask, einsum=einsum)

        out = self.fc(x)

        return out

class Transformer(nn.Module):
    def __init__(self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_dim=256,
            num_layers=6,
            forward_expansion=4,
            num_heads=8,
            dropout=0,
            device='cuda',
            max_length=100
            ):
        super().__init__()

        self.encoder = Encoder(
                src_vocab_size,
                embed_dim,
                num_layers,
                num_heads,
                device,
                forward_expansion,
                dropout,
                max_length)

        self.decoder = Decoder(
                trg_vocab_size,
                embed_dim,
                num_layers,
                num_heads,
                forward_expansion,
                dropout,
                device,
                max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        ### (bs,1,1,src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        bs, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
                bs, 1, trg_len, trg_len)

        return trg_mask.to(device)

    def forward(self, src, trg, einsum=True):
         src_mask = self.make_src_mask(src)
         trg_mask = self.make_trg_mask(trg)

         enc_out = self.encoder(src, src_mask)
         dec_out = self.decoder(trg, enc_out, src_mask, trg_mask, einsum)

         return dec_out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    num_layers = 1

    model = Transformer(src_vocab_size, 
                        trg_vocab_size, 
                        src_pad_idx, 
                        trg_pad_idx, 
                        num_layers=num_layers,
                        device=device).to(device)

    einsum = False
    out = model(x, trg[:, :-1], einsum=einsum)

    #out = model(x, trg[:, :-1])

    print(out.shape)


