import openglvoxelizer

vox = openglvoxelizer.Voxelizer()
vox.addMeshes({
    r'coil.stl':'print_body',
    r'core.stl':'attenuating_body',
    })

print_body = vox.voxelize('print_body',xy_voxel_size=0.01,z_voxel_size=0.1,voxel_value=1)
attenuating_body = vox.voxelize('attenuating_body',xy_voxel_size=0.01,z_voxel_size=0.1,voxel_value=2)

combined = print_body + attenuating_body

import numpy as np
np.save(r'combined_array.npy',combined)
np.save(r'print_body_array.npy',print_body)
np.save(r'attenuating_body_array.npy',attenuating_body)