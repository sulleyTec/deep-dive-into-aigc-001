import random
import matplotlib.pyplot as plt
import numpy as np

import torch
from transformer import Attention, TransformerBlock

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
                                    forward_expansion=forward_expansion).to('cuda')

    def forward(self, Q, K, V, mask):

        out = self.trans_perf(Q=Q, K=K, V=V, mask=mask, einsum=False)
        return out


def plot_arrays_of_points(dict_of_arrays, title, 
                          xy_name, save_path, 
                          x_coord, mode):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'lime', 'cyan', 'indigo', 'silver', 'gold', 'maroon', 'navy', 'teal']

    plt.figure(figsize=(10, 6))  # Set the size of the plot

    for idx, (key, points) in enumerate(dict_of_arrays.items()):
        color = colors[idx % len(dict_of_arrays)]  # Choose a color from the list

        # Extract x and y coordinates from points
        if mode=='seq':
            x_coords = [point[0] for point in points][1:]
            y_coords = [point[1] for point in points][1:]
        else:
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]

        # Plot the x and y coordinates with the chosen color and label
        plt.plot(x_coords, y_coords, color, label=key, marker='o')

    plt.xlabel(xy_name[0])
    plt.ylabel(xy_name[1])
    plt.legend()
    plt.title(title)

    # Customize x-axis ticks
    x_ticks = range(x_coord['min'], x_coord['max'], x_coord['step'])
    plt.xticks(x_ticks)

    # Save the plot as an image file
    plt.savefig(save_path)

def plot_stacked_bars_with_parts(scales, save_path, x_name):
    #labels = ['qkv_gemm', 'V@K', 'softmax@V', 'proj', 'FF0', 'FF1',
    #          'permute', '2multiheads', 'mask_fill', 'softmax',
    #          'reshape', 'ReLU', 'NormAdd']

    labels = ['qkv_gemm', 'attention', 'proj', 'FF0/FF1',
              'permute', '2multiheads', 'mask_fill', 'softmax',
              'reshape', 'ReLU', 'NormAdd']

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'lime', 'cyan', 'indigo', 'silver', 'gold', 'maroon', 'navy', 'teal']

    plt.figure(figsize=(10, 6))  # Set the size of the plot

    x_values = np.arange(len(scales))  # x coordinates for the bars
    for idx, scale in enumerate(scales):
        bottom = 0  # Initial bottom value for stacking

        for part_idx, value in enumerate(scale):
            if idx==0:
                plt.bar(x_values[idx], value, bottom=bottom, 
                        color=colors[part_idx], 
                        label=f"{labels[part_idx]}")
            else:
                plt.bar(x_values[idx], value, bottom=bottom, 
                        color=colors[part_idx])

            bottom += value  # Update the bottom value for stacking

        x_values[idx] += 0.5  # Shift x value for the next bar

    plt.xlabel(x_name)
    plt.ylabel("run time")
    plt.title("Ops runing time according to sequence length")
    plt.xticks(np.arange(len(scales)) + 0.2, range(1, len(scales)+1))
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.02))

    plt.savefig(save_path)


def scale_of_time(calc_Time, scale_save_path, mode):

    scale_list = [[0]*len(calc_Time) for _ in range(len(calc_Time['qkv_gemm']))]

    for i in range(len(calc_Time['qkv_gemm'])): # 17
        for j, (k, v) in enumerate(calc_Time.items()): # 13
            scale_list[i][j] = v[i][1]

    for i in range(len(scale_list)):
        s = sum(scale_list[i])
        for j in range(len(scale_list[i])):
            scale_list[i][j] /= s


    plot_stacked_bars_with_parts(scale_list, 
                                 scale_save_path, 
                                 x_name='seq_len*128' if mode=='seq' else 'embded_dim*128')

def performance(start, end, 
                MACs_title,
                Time_title,
                Memory_title,
                MACs_save_path,
                Time_save_path,
                Mem_save_path,
                scale_save_path,
                mode):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FP16 = FP32 = 34.1*1024*1024*1024 # GFLOPS
    band_width = 912. # GB/s

    step = 1024
    bs = 1
    seq_len = 512
    embed_dim = 256

    calc_MACs = {'qkv_gemm':[],
                 'score_mul':[],
                 'score_V_mul':[],
                 'proj':[],
                 'FF0':[],
                 'FF1':[]
                }

    '''
    calc_Time = {'qkv_gemm':[],
                 'score_mul':[],
                 'score_V_mul':[],
                 'proj':[],
                 'FF0':[],
                 'FF1':[],
                 'shape_permute':[],
                 'split_to_multiheads':[],
                 'mask_fill': [],
                 'softmax': [],
                 'permute_reshape':[],
                 'ReLU': [],
                 'NormAdd': []
                }
    '''
    calc_Time = {'qkv_gemm':[],
                 'score_mul':[],
                 'proj':[],
                 'FF0':[],
                 'shape_permute':[],
                 'split_to_multiheads':[],
                 'mask_fill': [],
                 'softmax': [],
                 'permute_reshape':[],
                 'ReLU': [],
                 'NormAdd': []
                }

    memory_perf = {'shape_permute':[],
                   'split_to_multiheads':[],
                   'mask_fill': [],
                   'softmax': [],
                   'permute_reshape':[],
                   'ReLU': [],
                   'NormAdd': []
                  }


    for i in range(start, end+step, step):
        print(f'{mode=}, {end=}, {i=}')
        ### Q.shape = K.shape = V.shape = (bs, seq_len, embed_dim)
        if mode=='seq':
            shape = (bs, i, embed_dim)
        elif mode == 'embed_dim':
            shape = (bs, seq_len, i)
        else:
            print(f'mode should be seq or embed_dim')
            return

        Q = torch.rand(shape).to(device)
        K = torch.rand(shape).to(device)
        V = torch.rand(shape).to(device)

        if mode=='seq':
            perf = Performance(embed_dim=embed_dim)
            mask = torch.ones(bs, 1, i, i).to(device)
        elif mode == 'embed_dim':
            perf = Performance(embed_dim=i)
            mask = torch.ones(bs, 1, seq_len, seq_len).to(device)

        out = perf.forward(Q, K, V, mask)

        ### Calc perf
        cmac1 = perf.trans_perf.attention.qkv_gemm.MACs/FP32
        cmac2 = perf.trans_perf.attention.score_mul.MACs/FP32
        cmac3 = perf.trans_perf.attention.score_V_mul.MACs/FP32
        cmac4 = perf.trans_perf.attention.proj.MACs/FP32
        cmac5 = perf.trans_perf.FF0.MACs/FP32
        cmac6 = perf.trans_perf.FF1.MACs/FP32

        calc_MACs['qkv_gemm'].append([i, cmac1])
        calc_MACs['score_mul'].append([i, cmac2])
        calc_MACs['score_V_mul'].append([i, cmac3])
        calc_MACs['proj'].append([i, cmac4])
        calc_MACs['FF0'].append([i, cmac5])
        calc_MACs['FF1'].append([i, cmac6])

        ctime1 = perf.trans_perf.attention.qkv_gemm.Time
        ctime2 = perf.trans_perf.attention.score_mul.Time
        ctime3 = perf.trans_perf.attention.score_V_mul.Time
        ctime4 = perf.trans_perf.attention.proj.Time
        ctime5 = perf.trans_perf.FF0.Time
        ctime6 = perf.trans_perf.FF1.Time

        ### Memory perf
        mmem1 = perf.trans_perf.attention.shape_permute.Memory/band_width
        mmem2 = perf.trans_perf.attention.split_to_multiheads.Memory/band_width
        mmem3 = perf.trans_perf.attention.mask_fill.Memory/band_width
        mmem4 = perf.trans_perf.attention.softmax.Memory/band_width
        mmem5 = perf.trans_perf.attention.permute_reshape.Memory/band_width
        mmem6 = perf.trans_perf.ReLU.Memory/band_width
        mmem7 = perf.trans_perf.NormAdd.Memory/band_width

        memory_perf['shape_permute'].append([i, mmem1])
        memory_perf['split_to_multiheads'].append([i, mmem2])
        memory_perf['mask_fill'].append([i, mmem3])
        memory_perf['softmax'].append([i, mmem4])
        memory_perf['permute_reshape'].append([i, mmem5])
        memory_perf['ReLU'].append([i, mmem6])
        memory_perf['NormAdd'].append([i, mmem7])

        mtime1 = perf.trans_perf.attention.shape_permute.Time
        mtime2 = perf.trans_perf.attention.split_to_multiheads.Time
        mtime3 = perf.trans_perf.attention.mask_fill.Time
        mtime4 = perf.trans_perf.attention.softmax.Time
        mtime5 = perf.trans_perf.attention.permute_reshape.Time
        mtime6 = perf.trans_perf.ReLU.Time
        mtime7 = perf.trans_perf.NormAdd.Time

        calc_Time['qkv_gemm'].append([i, ctime1])
        calc_Time['score_mul'].append([i, ctime2])
        #calc_Time['score_V_mul'].append([i, ctime3])
        calc_Time['proj'].append([i, ctime4])
        calc_Time['FF0'].append([i, ctime5])
        #calc_Time['FF1'].append([i, ctime6])
        calc_Time['shape_permute'].append([i, mtime1])
        calc_Time['split_to_multiheads'].append([i, mtime2])
        calc_Time['mask_fill'].append([i, mtime3])
        calc_Time['softmax'].append([i, mtime4])
        calc_Time['permute_reshape'].append([i, mtime5])
        calc_Time['ReLU'].append([i, mtime6])
        calc_Time['NormAdd'].append([i, mtime7])

    scale_of_time(calc_Time, scale_save_path, mode)

    # Call the function to plot the arrays of points
    x_coord = {'min':start, 'max': end, 'step':step}
    plot_arrays_of_points(calc_MACs, 
                          title=MACs_title,
                          xy_name=['seq len', 'MACs'], 
                          save_path=MACs_save_path,
                          x_coord=x_coord,
                          mode=mode)

    plot_arrays_of_points(calc_Time, 
                          title=Time_title,
                          xy_name=['seq len', 'Time'], 
                          save_path=Time_save_path,
                          x_coord=x_coord,
                          mode=mode)

    plot_arrays_of_points(memory_perf, 
                          title=Memory_title,
                          xy_name=['seq len', 'memory'], 
                          save_path=Mem_save_path,
                          x_coord=x_coord,
                          mode=mode)

if __name__ == '__main__':

    #seq_len_start = 128
    #embed_dim_start = 128
    seq_len_start = 1024
    embed_dim_start = 1024

    #max_seq_len = 16384
    max_seq_len = 11264
    max_embed_dim = 8192
    #max_embed_dim = 12288

    performance(start=seq_len_start, 
                end=max_seq_len, 
                MACs_title=f'Calc Op MACs: seq_len from {seq_len_start} to {max_seq_len}', 
                Time_title=f'Calc Op Time: seq_len from {seq_len_start} to {max_seq_len}', 
                Memory_title=f'Memory Op: seq_len from {seq_len_start} to {max_seq_len}', 
                MACs_save_path='seq_MACs.png',
                Time_save_path='seq_Time.png',
                Mem_save_path='seq_len_Mem.png',
                scale_save_path='seq_scale.png',
                mode='seq')

    performance(start=embed_dim_start, 
                end=max_embed_dim, 
                MACs_title=f'Calc Op MACs: embeded_dim from {embed_dim_start} to {max_embed_dim}', 
                Time_title=f'Calc Op Time: embeded_dim from {embed_dim_start} to {max_embed_dim}', 
                Memory_title=f'Memory Op: embed_dim from {embed_dim_start} to {max_embed_dim}', 
                MACs_save_path='embed_dim_MACs.png',
                Time_save_path='embed_dim_Time.png',
                Mem_save_path='embed_dim_Mem.png',
                scale_save_path='embed_dim_scale.png',
                mode='embed_dim')

    print('done')

