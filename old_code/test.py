import taichi as ti
import numpy as np
import tqdm

ti.init(arch=ti.gpu)
n = 512
vel = ti.Vector.field(2, float, (n, n))
dye = ti.field(float, (n, n))

@ti.kernel
def advect():
    for i, j in dye:
        x = ti.Vector([i, j]) - vel[i, j] * 0.1
        x = ti.max(0, ti.min(n - 1, x))
        dye[i, j] = dye[int(x[0]), int(x[1])]

@ti.kernel
def add_source():
    for i, j in dye:
        if (i - n // 2)**2 + (j - n // 2)**2 < 3000:
            dye[i, j] += 0.05
            vel[i, j] += ti.Vector([-0.5, 1.0])

gui = ti.GUI("Fluid Simulation", (n, n))
for t in tqdm.tqdm(range(1000)):
    add_source()
    advect()
    gui.set_image(np.power(dye.to_numpy(), 0.5))  # smooth color
    gui.show()
