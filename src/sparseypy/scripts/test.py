from sparseypy.core.model_layers.sparsey_layer import SparseyLayer, SparseyLayerOld, SparseyLayerV2

import torch
import time
import matplotlib.pyplot as plt

macs = [
    4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144,
    169, 196, 225, 256, 289, 324, 361, 400, 441, 
    484, 529, 576, 625, 676, 729, 784, 841, 900, 961,
    1024
]

batch_size = 1
num_reps = 25

v1s = []
v2s = []
v3s = []

for i, num_macs in enumerate(macs):
    print(f'\n---- num macs: {num_macs} ----')

    layer_3 = SparseyLayer(
        True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
        int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
        'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
        1.3, torch.device('cpu')
    )

    layer_2 = SparseyLayerV2(
        True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
        int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
        'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
        1.3, torch.device('cpu')
    )

    layer_1 = SparseyLayerOld(
        True, 'rect', num_macs, 5, 5, int(num_macs ** 0.5),
        int(num_macs ** 0.5), 0.4, 5, 5, 10, 10, 100,
        'rect', 0, 28.0, 5.0, 0.4, 10, 0.7, 0.2, 0.7, 0.2,
        1.3, torch.device('cpu')
    )

    #################

    x = torch.bernoulli(0.25 * torch.rand(batch_size, 100, 25)).float().cpu()
    start = time.time()

    for i in range(num_reps):
        out =  layer_3(x)

    print('v3:', (time.time() - start) / num_reps)
    v3s.append((time.time() - start) / num_reps)

    #################

    x = torch.bernoulli(0.25 * torch.rand(batch_size, 100, 25)).float().cpu()
    start = time.time()

    for i in range(num_reps):
        out =  layer_2(x)

    print('v2:', (time.time() - start) / num_reps)
    v2s.append((time.time() - start) / num_reps)

    #################

    x = torch.bernoulli(
        0.1 * torch.rand(
            batch_size, 10, 10, 5, 5)
        ).float().cpu()

    start = time.time()

    for i in range(num_reps):
        out =  layer_1(x)

    print('v1:', (time.time() - start) / num_reps)
    v1s.append((time.time() - start) / num_reps)

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.plot(macs, v1s, label='SparseyLayer V1 (macwise)')
ax.plot(macs, v2s, label='SparseyLayer V2 (layerwise)')
ax.plot(macs, v3s, label='SparseyLayer V3 (layerwise + new RFs)')

ax.scatter(macs, v1s)
ax.scatter(macs, v2s)
ax.scatter(macs, v3s)

ax.set_title('Forward pass latency (batch size 1, input 100x10x10)')
ax.set_xlabel('Number of MACs in layer')
ax.set_ylabel('Latency (s)')

ax.set_yscale('log')

ax.legend()

plt.show()