import torch
from trans import Attention, TransformerBlock
# QKV Gemm --> got
# split tensor into multi-heads --> got
# permute  --> got
# ScoreMul --> got
# Masked --> got
# Softmax --> got
# score@V --> got
# Projection --> got
# FFN0 --> got
# Relu --> got
# FFN1 --> got
# ADD & LayerNorm --> got

# calc: MACs, Memory

class Performance:
    def __init__(self, 
                 embed_dim=256,
                 num_heads=1,
                 dropout=0,
                 forward_expansion=4):

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.trans_perf = TransformerBlock(embed_dim, 
                                    num_heads,
                                    dropout=dropout,
                                    forward_expansion=forward_expansion)

    def draw_performance(self, num, points, opt, title, save_path, draw_opt=False):
        # Create a new figure
        plt.figure()

        # Plot the connected lines between points
        x_values, y_values = zip(*points)
        plt.plot(x_values, y_values, marker='o', color='blue', label='Points')

        # Find max x and y values
        max_x = max(x_values)
        max_y = max(y_values)

        # Customize the plot if needed (e.g., title, labels, grid, etc.)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()

        # Draw the 'opt' coordinate as text
        x, y = opt
        plt.text(0.95, 0.05, f"oneflow: ({x=}, {y=:.2f})", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Draw max x and y values as text
        plt.text(0.95, 0.1, f"{max_x=}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')
        plt.text(0.95, 0.15, f"{max_y=:.2f}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Draw the 'num' value as text
        plt.text(0.95, 0.2, f"#elements: {num}", transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='bottom', color='black')

        # Save the image
        plt.savefig(save_path)
        plt.close()

    def forward(self, Q, K, V, mask):

        out = self.trans_perf(Q=Q, K=K, V=V, mask=mask, einsum=False)
        return out

if __name__ == '__main__':

    FP16 = FP32 = 34.1*1024*1024*1024*1024 # TFLOPS
    band_width = 912. # GB/s
    #band_width = 912.*1024.*1024.*1024. # Bytes/s

    bs = 1
    #seq_len = 1
    #embed_dim = 1

    seq_len = 512
    embed_dim = 1024

    calc_perf = {'qkv_gemm':[],
                 'score_mul':[],
                 'score_V_mul':[],
                 'proj':[],
                 'FF0':[],
                 'FF1':[]
                }

    memory_perf = {'shape_permute':[],
                   'split_to_multiheads':[],
                   'mask_fill': [],
                   'softmax': [],
                   'permute_reshape':[],
                   'ReLU': [],
                   'NormAdd': []
                  }

    ### Q.shape = K.shape = V.shape = (bs, seq_len, embed_dim)
    shape = (bs, seq_len, embed_dim)

    Q = torch.rand(shape)
    K = torch.rand(shape)
    V = torch.rand(shape)

    mask = torch.ones(bs, 1, seq_len, seq_len)

    perf = Performance(embed_dim=embed_dim)
    out = perf.forward(Q, K, V, mask)

    ### Calc perf
    cmac1 = perf.trans_perf.attention.qkv_gemm.MACs/FP32
    cmac2 = perf.trans_perf.attention.score_mul.MACs/FP32
    cmac3 = perf.trans_perf.attention.score_V_mul.MACs/FP32
    cmac5 = perf.trans_perf.attention.proj.MACs/FP32
    cmac6 = perf.trans_perf.FF0.MACs/FP32
    cmac7 = perf.trans_perf.FF1.MACs/FP32

    calc_perf['qkv_gemm'].append([seq_len, cmac1])
    calc_perf['score_mul'].append([seq_len, cmac1])
    calc_perf['score_V_mul'].append([seq_len, cmac1])
    calc_perf['proj'].append([seq_len, cmac1])
    calc_perf['FF0'].append([seq_len, cmac1])
    calc_perf['FF1'].append([seq_len, cmac1])

    ctime1 = perf.trans_perf.attention.qkv_gemm.Time
    ctime2 = perf.trans_perf.attention.score_mul.Time
    ctime3 = perf.trans_perf.attention.score_V_mul.Time
    ctime5 = perf.trans_perf.attention.proj.Time
    ctime6 = perf.trans_perf.FF0.Time
    ctime7 = perf.trans_perf.FF1.Time

    ### Memory perf
    mmem1 = perf.trans_perf.attention.shape_permute.Memory/band_width
    mmem2 = perf.trans_perf.attention.split_to_multiheads.Memory/band_width
    mmem3 = perf.trans_perf.attention.mask_fill.Memory/band_width
    mmem4 = perf.trans_perf.attention.softmax.Memory/band_width
    mmem5 = perf.trans_perf.attention.permute_reshape.Memory/band_width
    mmem6 = perf.trans_perf.ReLU.Memory/band_width
    mmem7 = perf.trans_perf.NormAdd.Memory/band_width

    memory_perf['shape_permute'].append([seq_len, mmem1])
    memory_perf['split_to_multiheads'].append([seq_len, mmem2])
    memory_perf['mask_fill'].append([seq_len, mmem3])
    memory_perf['softmax'].append([seq_len, mmem4])
    memory_perf['permute_reshape'].append([seq_len, mmem5])
    memory_perf['ReLU'].append([seq_len, mmem6])
    memory_perf['NormAdd'].append([seq_len, mmem7])

    mtime1 = perf.trans_perf.attention.shape_permute.Time
    mtime2 = perf.trans_perf.attention.split_to_multiheads.Time
    mtime3 = perf.trans_perf.attention.mask_fill.Time
    mtime4 = perf.trans_perf.attention.softmax.Time
    mtime4 = perf.trans_perf.attention.permute_reshape.Time
    mtime5 = perf.trans_perf.ReLU.Time
    mtime6 = perf.trans_perf.NormAdd.Time

    print(f'input.shape={shape}, {out.shape}')


