'''
    by raymond
'''
import torch
import torch.nn as nn

class Perf:
    def __init__(self):
        self.MACs = 0.
        self.Memory = 0.
        self.Time = 0.

class Attention(nn.Module):
    ### Memory op: 
    ###     1. __shape_permute
    ###     2. __split_to_multiheads
    ###     3. __mask_fill
    ###     4. __softmax
    ###     5. __permute_reshape
    ###
    ### Calc op:
    ###     1. __qkv_gemm
    ###     2. __score_mul
    ###     3. __score_V_mul
    ###     4. __proj

    def __init__(self, embed_dim=512, num_heads=12):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        assert self.head_dim*num_heads==embed_dim, \
                'embed_dim must be divisable by num_heads'

        self.values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys= nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.fc = nn.Linear(self.head_dim*self.num_heads, self.embed_dim, bias=False)

        ### record params
        self.shape_permute = Perf()
        self.split_to_multiheads = Perf()
        self.mask_fill = Perf()
        self.softmax = Perf()

        self.qkv_gemm = Perf()
        self.score_mul = Perf()
        self.score_V_mul = Perf()
        self.permute_reshape = Perf()
        self.proj = Perf()

        ### init timing
        self.cuda_evnet_timing()

    def cuda_evnet_timing(self):
        # Create CUDA events
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize the GPU
        torch.cuda.synchronize()

    def __shape_permute(self):
        self.start_event.record()
        ### Q.shape = (bs, num_heads, q_seq_len, head_dim)
        self.Q = self.Q.permute(0,2,1,3)

        ### K.shape = (bs, num_heads, head_dim, k_seq_len)
        self.K = self.K.permute(0,2,3,1)

        ### V.shape = (bs, num_heads, v_seq_len=k_seq_len, head_dim)
        self.V = self.V.permute(0,2,1,3) 

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.shape_permute.Memory = self.Q.element_size()*\
                (self.Q.numel()+self.K.numel()+self.V.numel())
        self.shape_permute.Time = elapsed_time_ms


    def __score_mul(self):
        self.start_event.record()

        ### energy.shape = (bs, num_heads, q_seq_len, k_seq_len)
        self.energy = torch.matmul(self.Q, self.K)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.score_mul.MACs = 2*self.bs*self.num_heads* \
                            self.q_seq_len*self.head_dim*self.k_seq_len
        self.score_mul.Memory = self.Q.element_size()*(self.Q.numel()+ \
                                self.K.numel()+self.energy.numel())
        self.score_mul.Time = elapsed_time_ms

    def __score_V_mul(self):
        self.start_event.record()
        self.out = torch.matmul(self.attention, self.V) 
### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.score_V_mul.MACs = 2*self.V.element_size()*self.bs*self.num_heads*\
                                self.q_seq_len*self.v_seq_len*self.head_dim
        self.score_V_mul.Memory = self.V.element_size()* \
                (self.attention.numel()+self.V.numel()+ self.out.numel())
        self.score_V_mul.Time = elapsed_time_ms

    def __mask_fill(self, mask):
        if mask is not None:
            self.start_event.record()
            ### set a large negative value to those positions corresponding to mask=0
            ### when calculate softmax, these values will get to 0
            self.energy = self.energy.masked_fill(mask==0, float('-1e20'))

            ### record 
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
            self.mask_fill.Memory = self.energy.element_size()* \
                                    self.energy.numel()
            self.mask_fill.Time = elapsed_time_ms

    def __softmax(self):
        self.start_event.record()

        ### softmax operates on the 3rd dim which is the last dim, 
        ### after this, every row of energy get normalized by softmax
        ### attention.shape = (bs, num_heads, q_seq_len, k_seq_len)
        self.attention = torch.softmax(self.energy/(self.embed_dim ** 0.5), dim=3)
        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        self.softmax.Memory = self.energy.element_size()* \
                (self.energy.numel() + self.attention.numel())
        self.softmax.Time = elapsed_time_ms

    def __proj(self):
        self.start_event.record()
        out_pre_numel = self.out.numel()
        ### (bs, q_seq_len, embed_dim) @ (embed_dim, embed_dim)
        ### ---> (bs, q_seq_len, embed_dim) = input Q.shape
        self.out = self.fc(self.out)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.proj.MACs = self.bs*self.q_seq_len*self.embed_dim*self.embed_dim
        self.proj.Memory = self.Q.element_size()* \
                (out_pre_numel+self.out.numel()+self.embed_dim**2)
        self.proj.Time = elapsed_time_ms

    def __permute_reshape(self):
        self.start_event.record()

        ### out.shape = Q.shape = (bs, q_seq_len, num_heads, head_dim)
        self.out = self.out.permute(0,2,1,3).contiguous() 
        ### out.shape = (bs, q_seq_len, embed_dim)
        self.out = self.out.view(self.bs, self.out.shape[1], -1)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.permute_reshape.Memory = self.out.element_size()*self.out.numel()
        self.permute_reshape.Time = elapsed_time_ms

    ### input: (bs, seq_len, num_heads, head_dim)
    def mm_perf(self, mask):

        self.__shape_permute()

        #self.start_event.record()
        #energy = torch.matmul(Q,K)
        #self.end_event.record()
        #torch.cuda.synchronize()
        #elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        #print(f'attention mm time: {elapsed_time_ms}')

        self.__score_mul()
        self.__mask_fill(mask)
        self.__softmax()

        ### V.shape         = (bs, v_seq_len, num_heads, head_dim)
        ### attention.shape = (bs, num_heads, q_seq_len, k_seq_len=v_seq_len)
        ### k_seq_len == v_seq_len != q_seq_len
        self.__score_V_mul()
        self.__permute_reshape()
        self.__proj()

        return self.out

    def mm_einsum(self, mask):
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
        energy = torch.einsum('nqhd, nkhd->nhqk', [self.Q, self.K])

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
        out = torch.einsum('nhql,nlhd->nqhd',[attention, self.V]).contiguous()

        out = out.view(self.bs, out.shape[1], -1)
        out = self.fc(out)

        return out

    def __qkv_gemm(self):
        self.start_event.record()

        pre_Q_numel = self.Q.numel()
        pre_K_numel = self.K.numel()
        pre_V_numel = self.V.numel()

        ### generally k_seq_len and v_seq_len are not equal, 
        ### k_seq_len and v_seq_len are the same
        self.Q = self.queries(self.Q)
        self.K = self.keys(self.K)
        self.V = self.values(self.V)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.qkv_gemm.MACs = self.bs*\
                             (self.q_seq_len+self.k_seq_len+self.v_seq_len)*\
                             self.embed_dim*self.embed_dim*self.embed_dim

        self.qkv_gemm.Memory = self.V.element_size()*\
                (pre_Q_numel+pre_K_numel+pre_V_numel+\
                 self.Q.numel()+self.K.numel()+self.V.numel()+\
                 3*self.embed_dim**2)

        self.qkv_gemm.Time = elapsed_time_ms

    def __split_to_multiheads(self):
        self.start_event.record()

        ### split Q,K,V into multi-heads
        self.Q = self.Q.reshape(self.bs, self.q_seq_len, 
                                self.num_heads, self.head_dim)
        self.K = self.K.reshape(self.bs, self.k_seq_len, 
                                self.num_heads, self.head_dim)
        self.V = self.V.reshape(self.bs, self.v_seq_len, 
                                self.num_heads, self.head_dim)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.split_to_multiheads.Memory = self.V.element_size()*\
                (self.Q.numel()+self.K.numel()+self.V.numel())

        self.split_to_multiheads.Time = elapsed_time_ms

    ### V.shape = (bs, seq_len, embed_dim)
    def forward(self, Q, K, V, mask, einsum=True):

        self.Q = Q
        self.K = K
        self.V = V

        self.bs, _, self.embed_dim = Q.shape

        ### generally k_seq_len and v_seq_len are not equal, 
        ### k_seq_len and v_seq_len are the same
        self.q_seq_len = self.Q.shape[1]
        self.k_seq_len = self.K.shape[1]
        self.v_seq_len = self.V.shape[1]

        ### QKV MatrixMyl: 
        self.Q = self.Q.view(-1, self.embed_dim)
        self.K = self.K.view(-1, self.embed_dim)
        self.V = self.V.view(-1, self.embed_dim)

        self.__qkv_gemm()
        self.__split_to_multiheads()

        if einsum:
            return self.mm_einsum(mask)

        return self.mm_perf(mask)

class TransformerBlock(nn.Module):
    ### Memory op: 
    ###     1. __ReLU
    ###     2. __NormAdd
    ###
    ### Calc op:
    ###     1. __FF0
    ###     2. __FF1

    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super().__init__()

        self.forward_expansion = forward_expansion
        self.embed_dim = embed_dim

        self.attention = Attention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff0 = nn.Linear(embed_dim, forward_expansion*embed_dim)
        self.relu = nn.ReLU()
        self.ff1 = nn.Linear(forward_expansion*embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.FF0 = Perf()
        self.FF1 = Perf()
        self.ReLU = Perf()
        self.NormAdd = Perf()

        self.cuda_evnet_timing()

    def cuda_evnet_timing(self):
        # Create CUDA events
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize the GPU
        torch.cuda.synchronize()

    def __FF0(self, x):
        self.start_event.record()
        pre_x_numel = x.numel()

        x = self.ff0(x)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.FF0.MACs = self.attention.bs* \
                        self.attention.q_seq_len*self.forward_expansion* \
                        self.embed_dim*self.embed_dim
        self.FF0.Memory = x.element_size()*(x.numel() + \
                        self.embed_dim*self.embed_dim*self.forward_expansion + \
                        pre_x_numel)
        self.FF0.Time = elapsed_time_ms
        return x

    def __FF1(self, x):
        self.start_event.record()
        pre_x_numel = x.numel()

        x = self.ff1(x)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.FF1.MACs = self.attention.bs* \
                        self.attention.q_seq_len*self.forward_expansion* \
                        self.embed_dim*self.embed_dim
        self.FF1.Memory = x.element_size()*(x.numel() + \
                        self.embed_dim*self.embed_dim*self.forward_expansion + \
                        pre_x_numel)
        self.FF1.Time = elapsed_time_ms

        return x

    def __ReLU(self, x):
        self.start_event.record()

        x = self.relu(x)

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        self.ReLU.Memory = x.element_size()*x.numel()
        self.ReLU.Time = elapsed_time_ms

        return x

    def __NormAdd(self, x, y, num):
        self.start_event.record()

        if num == 1:
            z = self.dropout(self.layer_norm1(x+y))
        else:
            z = self.dropout(self.layer_norm2(x+y))

        ### record 
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

        self.NormAdd.Memory = z.element_size()*(z.numel()+x.numel()+y.numel())
        self.NormAdd.Time = elapsed_time_ms

        return z


    def forward(self, Q, K, V, mask, einsum=True):

        attention = self.attention(Q,K,V,mask, einsum)

        x = self.__NormAdd(attention, Q, 1)
        y = self.__FF0(x)
        y = self.__ReLU(y)
        y = self.__FF1(y)
        x = self.__NormAdd(x, y, 2)

        return x

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

    print(out.shape)


