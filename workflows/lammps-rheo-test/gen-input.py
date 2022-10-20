import numpy as np


data = np.linspace(-5.5, 5.5, 10)
ds = [7/6, 5/6]

dx = 0
dy = 0
dz = 0

i = 1
for x in data:
    for y in data:
        for z in data:
            # dx = 4*(np.random.rand() - 0.5)
            # dy = 4*(np.random.rand() - 0.5)
            # dz = 4*(np.random.rand() - 0.5)

            d = ds[i%2]
            
            print(i, i%2 + 1, d, 1.0, x + dx, y*2 + dy, z + dz)
            i += 1
