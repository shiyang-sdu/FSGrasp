import numpy as np
from thirdparty.binvox_rw import Voxels, write
from plyfile import PlyData, PlyElement


def load_vertices_from_ply(ply_filename):
    ply_data = PlyData.read(ply_filename)
    vertices = ply_data['vertex']

    vertex_data = np.stack([
        vertices['x'],
        vertices['y'],
        vertices['z']
    ], axis=-1)

    return vertex_data

def ply_to_binvox(ply_filename, hollow_output, solid_output, resolution=50):

    vertices = load_vertices_from_ply(ply_filename)

    voxels = np.zeros([resolution] * 3, dtype=np.bool)

    for vertex in vertices:
        x, y, z = (vertex * resolution).astype(int)
        voxels[x, y, z] = True


    hollow_voxels = Voxels(voxels, (1, 1, 1), (0, 0, 0), 1, 'xyz')
    # hollow_voxels.write(hollow_output)
    # with open(hollow_output,'w+') as ho_file:
    hollow_output_file = open(hollow_output, 'w+')
    write(hollow_voxels, hollow_output_file)

    solid_voxels = Voxels(np.ones([resolution] * 3, dtype=np.bool), (1, 1, 1), (0, 0, 0), 1, 'xyz')
    # solid_voxels.write(solid_output)
    solid_output_file = open(solid_output, 'w+')
    write(solid_voxels, solid_output_file)

if __name__ == '__main__':
    ply_filename = '/home/sy/Projects/3Dgrab/ContactPose-ML/data/object_models/mug.ply'
    hollow_output = '/home/sy/Projects/3Dgrab/ContactPose-ML/data/binvoxes/guitar_hollow.binvox'
    solid_output = '/home/sy/Projects/3Dgrab/ContactPose-ML/data/binvoxes/guitar_solid.binvox'
    ply_to_binvox(ply_filename, hollow_output, solid_output)
