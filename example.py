import openglvoxelizer

vox = openglvoxelizer.Voxelizer()
vox.addMeshes({
    r'coil.stl':'print_body',
    r'core.stl':'attenuating_body',
    })

print_body = vox.voxelize('print_body',0.01,1)
attenuating_body = vox.voxelize('attenuating_body',0.01,2)

import numpy as np
np.save(r'print_body_array.npy',print_body)
np.save(r'attenuating_body_array.npy',attenuating_body)