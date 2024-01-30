import numpy as np

'''
  ***  Signed distance field file is binary. Format:
    - resolutionX,resolutionY,resolutionZ (three signed 4-byte integers (all equal), 12 bytes total)
    - bminx,bminy,bminz (coordinates of the lower-left-front corner of the bounding box: (three double precision 8-byte real numbers , 24 bytes total)
    - bmaxx,bmaxy,bmaxz (coordinates of the upper-right-back corner of the bounding box: (three double precision 8-byte real numbers , 24 bytes total)
    - distance data (in single precision; data alignment: 
        [0,0,0],...,[resolutionX,0,0],
        [0,1,0],...,[resolutionX,resolutionY,0],
        [0,0,1],...,[resolutionX,resolutionY,resolutionZ]; 
    total num bytes: sizeof(float)*(resolutionX+1)*(resolutionY+1)*(resolutionZ+1))
    - closest point for each grid vertex(3 coordinates in single precision)
'''


def create_sdf(b_min, b_max, resX, resY, resZ):
    coords = np.mgrid[:resX, :resY, :resZ]
    IND = np.eye(4)
    length = b_max - b_min
    IND[0, 0] = length[0] / resX
    IND[1, 1] = length[1] / resY
    IND[2, 2] = length[2] / resZ
    IND[0:3, 3] = b_min
    return coords, IND


def load_sdf(file_path, read_closest_points=False, verbose=False):
    '''
    :param file_path: file path
    :param read_closest_points: whether to read closest points for each grid vertex
    :param verbose: verbose flag
    :return:
        b_min: coordinates of the lower-left-front corner of the bounding box
        b_max: coordinates of the upper-right-back corner of the bounding box
        volume: distance data in shape (resolutionX+1)*(resolutionY+1)*(resolutionZ+1)
        closest_points: closest points in shape (resolutionX+1)*(resolutionY+1)*(resolutionZ+1)
    '''
    with open(file_path, 'rb') as fp:

        res_x = int(np.fromfile(fp, dtype=np.int32,
                                count=1))  # note: the dimension of volume is (1+res_x) x (1+res_y) x (1+res_z)
        res_x = - res_x
        res_y = -int(np.fromfile(fp, dtype=np.int32, count=1))
        res_z = int(np.fromfile(fp, dtype=np.int32, count=1))
        if verbose: print("resolution: %d %d %d" % (res_x, res_y, res_z))

        b_min = np.zeros(3, dtype=np.float64)
        b_min[0] = np.fromfile(fp, dtype=np.float64, count=1)
        b_min[1] = np.fromfile(fp, dtype=np.float64, count=1)
        b_min[2] = np.fromfile(fp, dtype=np.float64, count=1)
        if verbose: print("b_min: %f %f %f" % (b_min[0], b_min[1], b_min[2]))

        b_max = np.zeros(3, dtype=np.float64)
        b_max[0] = np.fromfile(fp, dtype=np.float64, count=1)
        b_max[1] = np.fromfile(fp, dtype=np.float64, count=1)
        b_max[2] = np.fromfile(fp, dtype=np.float64, count=1)
        if verbose: print("b_max: %f %f %f" % (b_max[0], b_max[1], b_max[2]))

        grid_num = (1 + res_x) * (1 + res_y) * (1 + res_z)
        volume = np.fromfile(fp, dtype=np.float32, count=grid_num)
        volume = volume.reshape(((1 + res_z), (1 + res_y), (1 + res_x)))
        volume = np.swapaxes(volume, 0, 2)
        if verbose: print("loaded volume from %s" % file_path)

        closest_points = None
        if read_closest_points:
            closest_points = np.fromfile(fp, dtype=np.float32, count=grid_num * 3)
            closest_points = closest_points.reshape(((1 + res_z), (1 + res_y), (1 + res_x), 3))
            closest_points = np.swapaxes(closest_points, 0, 2)

    return b_min, b_max, volume, closest_points
