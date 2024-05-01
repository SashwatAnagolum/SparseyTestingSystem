"""
test.py - short CUDA testing script
"""
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer, SparseyLayerOld, SparseyLayerV2

import torch
import time
import matplotlib.pyplot as plt

device = torch.device('cuda')

# macs = [
#     4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144,
#     169, 196, 225, 256, 289, 324, 361, 400, 441, 
#     484, 529, 576, 625, 676, 729, 784, 841, 900, 961,
#     1024
# ]

# batch_size = 1
# num_reps = 25

# v1s = []
# v2s = []
# v3s = []

# for i, num_macs in enumerate(macs[:4]):
#     print(f'\n---- num macs: {num_macs} ----')

#     layer_3 = SparseyLayer(
#         True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
#         int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
#         'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
#         1.3, device
#     )

#     layer_2 = SparseyLayerV2(
#         True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
#         int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
#         'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
#         1.3, device
#     )

#     layer_1 = SparseyLayerOld(
#         True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
#         int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
#         'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
#         1.3, device
#     )

#     #################

#     x = torch.bernoulli(0.25 * torch.rand(batch_size, 100, 25)).float().to(device)
#     start = time.time()

#     for i in range(num_reps):
#         out =  layer_3(x)

#     print('v3:', (time.time() - start) / num_reps)
#     v3s.append((time.time() - start) / num_reps)

#     #################

#     x = torch.bernoulli(0.25 * torch.rand(batch_size, 100, 25)).float().to(device)
#     start = time.time()

#     for i in range(num_reps):
#         out =  layer_2(x)

#     print('v2:', (time.time() - start) / num_reps)
#     v2s.append((time.time() - start) / num_reps)

#     #################

#     x = torch.bernoulli(
#         0.1 * torch.rand(
#             batch_size, 10, 10, 5, 5)
#     ).float().to(device)

#     start = time.time()

#     for i in range(num_reps):
#         out =  layer_1(x)

#     print('v1:', (time.time() - start) / num_reps)
#     v1s.append((time.time() - start) / num_reps)

# fig, ax = plt.subplots(1, 1, figsize=(20, 10))

# ax.plot(macs, v1s, label='SparseyLayer V1 (macwise)')
# ax.plot(macs, v2s, label='SparseyLayer V2 (layerwise)')
# ax.plot(macs, v3s, label='SparseyLayer V3 (layerwise + new RFs)')

# ax.scatter(macs, v1s)
# ax.scatter(macs, v2s)
# ax.scatter(macs, v3s)

# ax.set_title('Forward pass latency (batch size 1, input 100x10x10)')
# ax.set_xlabel('Number of MACs in layer')
# ax.set_ylabel('Latency (s)')

# ax.set_yscale('log')

# ax.legend()

# plt.savefig('test.png', transparent=True)

# num_macs = 256

# layer_3 = SparseyLayer(
#     True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
#     int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
#     'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
#     1.3, device
# )

# batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# reps = 100

# times = []

# for bsz in batch_sizes:
#     total = 0.0

#     for i in range(reps):  
#         x = torch.bernoulli(0.25 * torch.rand(bsz, 100, 25)).float().to(device)

#         start = time.time()
#         layer_3(x)
#         total += time.time() - start

#     times.append(total)

# print(times)

tps = [308.0101810768853, 2106.806440513654, 4389.103380798279, 8773.042732539898, 14373.314999603768, 20072.55194320607, 29065.674550754266, 34943.00135574996, 37236.13119979803, 38556.95606897394, 39700.08961473797, 39767.173656468076]

fig, ax = plt.subplots(1,1, figsize=(10, 5))

ax.bar(range(12), height=tps)
ax.set_xticks([i for i in range(12)], [str(int(2 ** i)) for i in range(12)])

ax.set_title('Eval throughput (items / s) | 256x10x10 layer | 100x10x10 input')
ax.set_xlabel('Batch size')
ax.set_ylabel('Throughput')

plt.show()