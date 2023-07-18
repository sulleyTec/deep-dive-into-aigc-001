import numpy as np
import torch
import torch.nn as nn
import libgfinfer as gf

#def numpy_mm(A, B):
#    if len(A.shape)<2 or len(B.shape)<2:
#        print('matrix dimension must >= 2')
#        return 
#
#    result_shape = []
#    for i in range(len(A.shape)-2):
#        result_shape.append(A.shape[i])
#
#    result_shape.append(A.shape[-2])
#    result_shape.append(B.shape[-1])
#
#    # Reshape A and B to 2-dimensional arrays
#    A_2d = np.reshape(A, (-1, A.shape[-2], A.shape[-1]))
#    B_2d = np.reshape(B, (-1, B.shape[-2], B.shape[-1]))
#
#    # Perform matrix multiplication
#    result_2d = np.matmul(A_2d, B_2d)
#
#    # Reshape the result back to the original shape
#    result = np.reshape(result_2d, tuple(result_shape))
#
#    return result
#
#### TESTS/SUBMISSION CODE FOR add()
#def test_mm():
#    # test 2 dim matrices 
#    m1 = np.random.rand(16, 64)
#    m2 = np.random.rand(64, 8)
#
#    numpy_result = numpy_mm(m1, m2)
#    gf_result = gf.matrix_multiply(m1, m2)
#    print(f'{numpy_result=}')
#    print(f'{gf_result=}')
#
#    assert gf_result.shape == (16,8)
#    np.testing.assert_allclose(gf_result, numpy_result, rtol=1e-5)
#
#    # test 3 dim matrices 
#    m1 = np.random.rand(8, 128, 64)
#    m2 = np.random.rand(8, 64, 16)
#
#    numpy_result = numpy_mm(m1, m2)
#    gf_result = gf.matrix_multiply(m1, m2)
#
#    assert gf_result.shape == (8,128,16)
#    np.testing.assert_allclose(gf_result, numpy_result, rtol=1e-5)
#
#    # test 4 dim matrices 
#    m1 = np.random.rand(8, 12, 32, 64)
#    m2 = np.random.rand(8, 12, 64, 16)
#
#    numpy_result = numpy_mm(m1, m2)
#    gf_result = gf.matrix_multiply(m1, m2)
#    assert gf_result.shape == (8,12,32,16)
#    np.testing.assert_allclose(gf_result, numpy_result, rtol=1e-5)
#
#
#def test_permute():
#    arr = np.random.rand(4,8,32,16)
#    brr = torch.from_numpy(arr)
#
#    perm = [1, 0, 3, 2]
#
#    arr = gf.permute(perm, arr)
#    brr = brr.permute(perm).numpy()
#
#    assert arr.shape == brr.shape
#    np.testing.assert_allclose(brr, arr)
#
##test_mm()
#test_permute()



#### BatchNorm2d
#m = nn.BatchNorm2d(100)
## Without Learnable Parameters
## 100 is channels
##m = nn.BatchNorm2d(100, affine=False)
#input = torch.randn(20, 100, 35, 45)
#output = m(input)
#ic0 = input[:,0,:,:].reshape(-1)
#oc0 = output[:,0,:,:].reshape(-1)
#
#imean = torch.mean(ic0)
#ivar = torch.var(ic0)
#eps = 1e-5
#
#gamma = m.weight # shape=()
#beta = m.bias # shape=()
#
#print(f'{gamma.shape=}')
#print(f'{beta.shape=}')
#out = (ic0 - imean)/torch.sqrt(ivar+eps)
#print(f'{torch.sum(oc0-out)=}')
#np.testing.assert_allclose(oc0.numpy(), out.numpy(), rtol=1e-4)


## LayerNorm
#N, C, H, W = 20, 5, 20, 10
#input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
#layer_norm = nn.LayerNorm([C, H, W])
#output = layer_norm(input)
#out = torch.zeros_like(input)
#
#eps = 1e-5
#
#gamma = layer_norm.weight
#beta = layer_norm.bias
#
#print(f'{input.shape=}')
#print(f'{gamma.shape=}')
#print(f'{beta.shape=}')
#
#imean = input.mean((1,2,3))
#ivar = input.var((1,2,3))
#eps = 1e-5
#
#for i in range(input.shape[0]):
#    #out[i] = (input[i] - imean[i])/torch.sqrt(ivar[i]+eps)*gamma + beta
#    out[i] = torch.div(input[i] - imean[i], torch.sqrt(ivar[i]+eps))*gamma + beta
#
#np.testing.assert_allclose(output.detach().numpy(), out.detach().numpy(), rtol=1e-3)
#
#print()
#


def test_mean():
    print('mmmmmmmmmmmmmmmmmmmmmm')
    N, C, H, W = 20, 5, 20, 10
    #N, C, H, W = 2,1,3,4
    input = torch.randn(N, C, H, W)
    #input = torch.arange(N*C*H*W, dtype=float).reshape(N,C,H,W)

    numpy_input = input.numpy()

    #reduce_axes = [0,-1]
    reduce_axes = [0]
    gf_out = gf.mean(reduce_axes, numpy_input)
    #torch_out = input.mean((0,-1)).detach().numpy()
    torch_out = input.mean((0)).detach().numpy()

    print(f'{input.shape=}')
    print(f'{torch_out.shape=}')
    print(f'{numpy_out.shape=}')

    #print(f'{torch_out=}')
    #print(f'{numpy_out=}')

    np.testing.assert_allclose(gf_out, torch_out, rtol=1e-3)

def test_var():
    print('vvvvvvvvvvvvvvvvvvvvv')
    N, C, H, W = 20, 5, 20, 10
    #N, C, H, W = 2,1,3,4
    input = torch.randn(N, C, H, W)
    #input = torch.arange(N*C*H*W, dtype=float).reshape(N,C,H,W)

    numpy_input = input.numpy()

    #reduce_axes = [0,-1]
    reduce_axes = [0]
    gf_out = gf.var(reduce_axes, numpy_input)
    #torch_out = input.var((0,-1)).detach().numpy()
    torch_out = input.var((0)).detach().numpy()

    print(f'{input.shape=}')
    print(f'{torch_out.shape=}')
    print(f'{numpy_out.shape=}')

    np.testing.assert_allclose(gf_out, torch_out, rtol=1e-5)


def test_broadcast():
    arr = np.zeros((6,4,2,3), dtype=float)
    brr = np.random.rand(4,1,3)

    crr = gf.broadcast(arr.shape, brr)
    drr = arr + brr
    print(f'{crr.shape=}')
    print(f'{drr.shape=}')
    np.testing.assert_allclose(crr, drr)

def test_element_wise_mul():
    arr = np.random.rand(4,2,3)
    brr = np.random.rand(1,3)

    numpy_result=np.multiply(brr, arr)
    gf_result = gf.element_wise_mul(arr, brr)

    print(f'{numpy_result.shape=}')

    np.testing.assert_allclose(gf_result, numpy_result)

def test_element_wise_add():
    print('test_element_wise_add')
    arr = np.random.rand(4,2,3)
    brr = np.random.rand(1,3)

    numpy_result=arr+brr
    gf_result = gf.element_wise_add(arr, brr)

    print(f'{numpy_result.shape=}')

    np.testing.assert_allclose(gf_result, numpy_result)

def test_element_wise_div():
    arr = np.random.rand(4,2,3)
    brr = np.random.rand(2,3)

    numpy_result=arr/brr
    gf_result = gf.element_wise_div(arr, brr)

    print(f'{numpy_result.shape=}')

    np.testing.assert_allclose(gf_result, numpy_result)

def test_element_wise_minus():
    arr = np.random.rand(4,2,3)
    brr = np.random.rand(2,3)

    numpy_result=arr-brr
    gf_result = gf.element_wise_minus(arr, brr)

    print(f'{numpy_result.shape=}')

    np.testing.assert_allclose(gf_result, numpy_result, rtol=1e-5)

def test_element_wise_sqrt():
    N, C, H, W = 20, 5, 20, 10
    arr = np.random.rand(N,C,H,W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    numpy_result=np.sqrt(arr)
    gf_result=gf.element_wise_sqrt(arr)

    print(f'{numpy_result.shape=}')
    print(f'{gf_result.shape=}')

    np.testing.assert_allclose(gf_result, numpy_result)

#test_element_wise_mul()
#test_element_wise_add()
#test_element_wise_div()
#test_element_wise_minus()
#test_element_wise_sqrt()

## LayerNorm
def test_layernorm():
    N, C, H, W = 20, 5, 20, 10
    input = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = nn.LayerNorm([C, H, W])

    numpy_result = layer_norm(input).detach().numpy()
    gf_result = gf.layernorm(input.detach().numpy())

    #print(f'{numpy_result.shape=}')
    #print(f'{gf_result.shape=}')

    np.testing.assert_allclose(gf_result, numpy_result)

#test_layernorm()

#test_mean()
#test_var()

def test_test():
    N, C, H, W = 20, 5, 20, 10
    input = torch.randn(N, C, H, W)

    reduce_axes = [0]
    gf_input = input.numpy()

    gf_mean = gf.mean(reduce_axes, gf_input)
    torch_mean = input.mean((0))
    np.testing.assert_allclose(gf_mean, torch_mean, rtol=1e-3)
    print(f'mean pass')

    gf_var = gf.var(reduce_axes, gf_input, 0)
    #torch_var = input.var((0))
    torch_var = torch.var(input, dim=0, unbiased=False)

    #print(f'{gf_var.shape=}')
    #print(f'{torch_var.shape=}')

    np.testing.assert_allclose(gf_var, torch_var, rtol=1e-5)
    print(f'var pass')

    gf_tmp1 = gf.element_wise_minus(gf_input, gf_mean)
    torch_tmp1 = input - torch_mean
    np.testing.assert_allclose(gf_tmp1, torch_tmp1, rtol=1e-3)
    print(f'tmp1 pass')

    gf_tmp2 = gf.element_wise_sqrt(gf_var)
    torch_tmp2 = torch_var.sqrt()
    np.testing.assert_allclose(gf_tmp2, torch_tmp2, rtol=1e-5)
    print(f'tmp2 pass')

    gf_tmp3 = gf.element_wise_div(gf_tmp1, gf_tmp2)
    torch_tmp3 = torch_tmp1/torch_tmp2
    np.testing.assert_allclose(gf_tmp3, torch_tmp3, rtol=1e-3)
    print(f'tmp3 pass')

    #layer_norm = nn.LayerNorm([C, H, W])
    layer_norm = nn.LayerNorm([W])
    torch_result = layer_norm(input).detach().numpy()

    # N, C, H, W = 20, 5, 20, 10
    tmp1 = input-input.mean(3).reshape(N,C,H,1)
    tmp2 = torch.sqrt(input.var(3, unbiased=False).reshape(N,C,H,1)+1e-3)

    calc_result = tmp1/tmp2

    print(f'{torch_result.shape=}')
    print(f'{calc_result.shape=}')

    np.testing.assert_allclose(torch_result, 
                               calc_result.detach().numpy(),
                               rtol=1e-2)

    #print(f'{input.shape=}')
    #print(f'{torch_out.shape=}')
    #print(f'{numpy_out.shape=}')

#test_test()
test_element_wise_add()


