# OpenGL Voxelizer

This is a standalone voxelizer based on OpenGL

## Requirements
- numpy
- [tqdm](https://anaconda.org/conda-forge/tqdm)
- [pyopengl](https://anaconda.org/anaconda/pyopengl)
- [pyglet](https://anaconda.org/conda-forge/pyglet)
- [trimesh](https://anaconda.org/conda-forge/trimesh)


## Example

The voxelizer uses the .stl file coordinates to determine the position of the voxelized mesh in the voxel array. When using the voxelizer for VAM, the mesh should be oriented in CAD space such that the axis which is to be along the rotation axis of the vial is the z-axis.
The minimum bound of an object should be equal to zero in the z-axis, e.g., `min(mesh_z_coordinates)=0`. 
The x and y position in space is also considered. This is especially important when more 2 or more meshes that are somehow related in physical space are voxelized. 
Ideally, the center of the mesh should be near `(x,y) = (0,0)` and `min(mesh_z_coordinates)=0`. This will ensure that the voxel array size is not expanded to fit meshes that are located at a large `(x,y)` offset from the origin.
``` python
import openglvoxelizer

vox = openglvoxelizer.Voxelizer()
vox.addMeshes({
    r'coil.stl':'print_body',
    r'core.stl':'attenuating_body',
    })

print_body = vox.voxelize('print_body',0.01,1)
attenuating_body = vox.voxelize('attenuating_body',0.01,2)
```

The voxel arrays can be saved to the disk with numpy
``` python
import numpy as np
np.save(r'print_body_array.npy',print_body)
np.save(r'attenuating_body_array.npy',attenuating_body)
```
