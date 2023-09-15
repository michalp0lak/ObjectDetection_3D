import numpy as np
import numba

################################################################################
#IOU

@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


# Voxelizer
class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_voxel_points,
                 max_voxels):

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_voxel_points = max_voxel_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size


    def generate(self, points, max_voxels, cloud_range, reflectance_sampling):

        return points_to_voxel(
            points, self._voxel_size, cloud_range,
            self._max_voxel_points, max_voxels, reflectance_sampling)


    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size

class CustomVoxelGenerator:
    def __init__(self,
                 voxel_size,
                 max_voxel_points,
                 reflectance_sampling):

        self._voxel_size = np.array(voxel_size, dtype=np.float32)
        self._max_voxel_points = max_voxel_points
        self._reflectance_sampling = reflectance_sampling


    def generate(self, points, point_cloud_range, max_voxels):

        return points_to_voxel(
            points, self._voxel_size, point_cloud_range,
            self._max_voxel_points, max_voxels, self._reflectance_sampling)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_voxel_points


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points,
                     max_voxels,
                     reflectance_sampling):

    """convert points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    
    # don't create large array in jit(nopython=True) code.
    voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)

    if reflectance_sampling:
        voxel_num = points_to_voxel_reflectance_kernel(
            points, voxel_size, coors_range, max_points,
            max_voxels, voxels, coors, num_points_per_voxel,
            coor_to_voxelidx)
    else:
        voxel_num = points_to_voxel_kernel(
            points, voxel_size, coors_range, max_points,
            max_voxels, voxels, coors, num_points_per_voxel,
            coor_to_voxelidx)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
 
    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            max_points,
                            max_voxels,
                            voxels,
                            coors,
                            num_points_per_voxel,
                            coor_to_voxelidx
                           ):

    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance

    N = points.shape[0]
    np.random.shuffle(points)
    
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0

    # Spatial check - is point in point cloud range
    failed = False

    for i in range(N):
        failed = False
        for j in range(ndim):
            # Get voxel index in each dimension for examined point
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            # Is out of grid?
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            #Assign index
            coor[j] = c
        if failed:
            continue

        # Get voxel ID from voxel-coord mask
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]    
        # If voxel was not assigned
        if voxelidx == -1:
             # Assign voxel id value
            voxelidx = voxel_num
            # If there is too much voxels -> break
            if voxel_num >= max_voxels:
                break
            # else increase voxel ID value by 1 and assign ID into voxel-coord mask
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # Assign voxel 3D coords
            coors[voxelidx] = coor

        # Get voxel-point size of given voxel
        num = num_points_per_voxel[voxelidx]    
        # If limit of points was not overreached

        if num < max_points:

            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num

@numba.jit(nopython=True)
def points_to_voxel_reflectance_kernel(points,
                            voxel_size,
                            coors_range,
                            max_points,
                            max_voxels,
                            voxels,
                            coors,
                            num_points_per_voxel,
                            coor_to_voxelidx
                           ):

    N = points.shape[0]
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0

    points = points[points[:, 3].argsort()[::-1]]
   
    # Spatial check - is point in point cloud range
    failed = False

    for i in range(N):

        #i = np.random.choice(range(points.shape[0]), 1, p=points[:,4])
        failed = False

        for j in range(ndim):
            # Get voxel index in each dimension for examined point
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            # Is out of grid?
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            #Assign index
            coor[j] = c
        if failed:
            continue

        # Get voxel ID from voxel-coord mask
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]    
        # If voxel was not assigned
        if voxelidx == -1:
             # Assign voxel id value
            voxelidx = voxel_num
            # If there is too much voxels -> break
            if voxel_num >= max_voxels:
                break
            # else increase voxel ID value by 1 and assign ID into voxel-coord mask
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # Assign voxel 3D coords
            coors[voxelidx] = coor

        # Get voxel-point size of given voxel
        num = num_points_per_voxel[voxelidx]    
        # If limit of points was not overreached

        if num < max_points:

            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num