import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

# BBOX MANIPULATIONS
#######################################################################
def bbox2corners3D(bbxs):

    # Define axis aligned vertices
    p_0 = np.concatenate(((bbxs[...,0]-bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]).reshape(-1,1)), axis=1) 

    p_1 = np.concatenate(((bbxs[...,0]+bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]).reshape(-1,1)), axis=1) 
                    
    p_2 = np.concatenate(((bbxs[...,0]+bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]).reshape(-1,1)), axis=1) 
                    
    p_3 = np.concatenate(((bbxs[...,0]-bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]).reshape(-1,1)), axis=1) 

    p_4 = np.concatenate(((bbxs[...,0]-bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]+bbxs[...,5]).reshape(-1,1)), axis=1) 

    p_5 = np.concatenate(((bbxs[...,0]+bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]+bbxs[...,5]).reshape(-1,1)), axis=1) 

    p_6 = np.concatenate(((bbxs[...,0]+bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]+bbxs[...,5]).reshape(-1,1)), axis=1) 

    p_7 = np.concatenate(((bbxs[...,0]-bbxs[...,3]*0.5).reshape(-1,1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).reshape(-1,1), 
                    (bbxs[...,2]+bbxs[...,5]).reshape(-1,1)), axis=1) 

    vertices = np.concatenate([p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7], axis = 1).reshape(-1,8,3)
    
    # Define center
    center = np.concatenate([bbxs[...,0].reshape(-1,1),bbxs[...,1].reshape(-1,1),bbxs[...,2].reshape(-1,1)], axis = -1)
    center = center.reshape(-1,1,3).repeat(8, axis = 1)

    #Define rotation
    x1 = np.concatenate([np.ones((bbxs.shape[0],1)), 
                         np.zeros((bbxs.shape[0],1)), 
                         np.zeros((bbxs.shape[0],1))], 
                         axis = 1)

    x2 = np.concatenate([np.zeros((bbxs.shape[0],1)), 
                         np.cos(bbxs[...,6].reshape(-1,1)), 
                         -np.sin(bbxs[...,6].reshape(-1,1))],
                         axis = 1)

    x3 = np.concatenate([np.zeros((bbxs.shape[0],1)), 
                         np.sin(bbxs[...,6].reshape(-1,1)), 
                         np.cos(bbxs[...,6].reshape(-1,1))],
                         axis = 1)

    rot_x = np.concatenate([x1,x2,x3], axis = 1).reshape(-1,3,3)

    y1 = np.concatenate([np.cos(bbxs[...,7].reshape(-1,1)), 
                         np.zeros((bbxs.shape[0],1)), 
                         np.sin(bbxs[...,7].reshape(-1,1))],
                         axis = 1)

    y2 = np.concatenate([np.zeros((bbxs.shape[0],1)), 
                         np.ones((bbxs.shape[0],1)), 
                         np.zeros((bbxs.shape[0],1))],
                         axis = 1)

    y3 = np.concatenate([-np.sin(bbxs[...,7].reshape(-1,1)), 
                         np.zeros((bbxs.shape[0],1)), 
                         np.cos(bbxs[...,7].reshape(-1,1))],
                         axis = 1)

    rot_y = np.concatenate([y1,y2,y3], axis = 1).reshape(-1,3,3)

    z1 = np.concatenate([np.cos(bbxs[...,8].reshape(-1,1)), 
                         -np.sin(bbxs[...,8].reshape(-1,1)),
                         np.zeros((bbxs.shape[0],1))],
                         axis = 1)

    z2 = np.concatenate([np.sin(bbxs[...,8].reshape(-1,1)), 
                         np.cos(bbxs[...,8].reshape(-1,1)), 
                         np.zeros((bbxs.shape[0],1))],
                         axis = 1)

    z3 = np.concatenate([np.zeros((bbxs.shape[0],1)), 
                         np.zeros((bbxs.shape[0],1)), 
                         np.ones((bbxs.shape[0],1))],
                         axis = 1)

    rot_z = np.concatenate([z1,z2,z3], axis = 1).reshape(-1,3,3)

    rot = np.matmul(np.matmul(rot_z[:,...], rot_y[:,...]), rot_x[:,...])
    centered_verts = vertices - center

    return np.matmul(centered_verts[:,...], rot.transpose(0,2,1)[:,...]) + center


# POINT CLOUD AUGMENTATIOS
#######################################################################
############################################################################
#Section of operations without random effect

def global_outlier_check(point_cloud: np.ndarray):

    norm = np.sum((point_cloud[:,:3] - np.mean(point_cloud[:,:3], axis=0))**(2),axis=1)**(0.5)

    return point_cloud[norm < np.mean(norm) + 5*np.std(norm),:]

def recenter(points):

    """Recenter pointcloud/features to origin.
    Typically used before rotating the pointcloud.
    Args:
        points: Pointcloud with or without features.
    """

    point_cloud_shift = points[:,:3].mean(0)

    points[:,:3] = points[:,:3] - point_cloud_shift
    
    return points, point_cloud_shift


def normalize(points, method):

    """Normalize pointcloud and/or (spatial/all) features.
    Points are normalized in [0, 1] and features can take custom
    scale and bias.

    Args:
        poins: Pointcloud.
    """

    if method == 'spatial':

        points[:,:3] -= points[:,:3].mean(0)
        points[:,:3] /= (points[:,:3].max(0) - points[:,:3].min(0))


    elif method == 'all':

        points[:,:-1] -= points[:,:-1].mean(0)
        points[:,:-1] /= (points[:,:-1].max(0) - points[:,:-1].min(0))

    else:
        raise ValueError(f"Unsupported method : {method}")

    return points

def vertical_cropper(points, vertical_range):

    """Sampler which crops point cloud to specified vertical range.
    Args:
        points: Pointcloud to augment.
        vertical_range: List of two values [z_min, z_max], which defines croped vertical range
                        above point cloud z_min value.
    """

    return points[np.where((points[:,2] > points[:,2].min()+1) & (points[:,2] < points[:,2].min()+7))[0],:]

############################################################################
#Section of operations with random effect

def PointShuffle(points, rng):

    """Shuffle point cloud points.
    Args:
        points: Pointcloud to augment.
        rng: Default random number generator initialized and shared in SegmantationAugmentor class.
    """

    shuffle_index = list(range(points.shape[0]))
    rng.shuffle(shuffle_index)

    return points[shuffle_index]

def rotate(points, rot_limits, method, rng):

    """Rotate the pointcloud.
    Two methods are supported. `vertical` rotates the pointcloud
    along yaw. `all` randomly rotates the pointcloud in all directions.
    Args:
        points: Pointcloud to augment.
        rot_limits: List of rotation limits for each XYZ coordinates in a form
                    [x_min_rot, y_min_rot, z_min_rot, x_max_rot, y_max_rot, z_max_rot]
        method: Method determines if 'all' rotations should be performed or just 'vertical' (around z-axis)
        rng: Default random number generator initialized and shared in SegmantationAugmentor class.
    """

    rotations = [np.deg2rad(rng.uniform(rot_limits[0], rot_limits[3])), 
                 np.deg2rad(rng.random.uniform(rot_limits[1], rot_limits[4])), 
                 np.deg2rad(rng.random.uniform(rot_limits[2], rot_limits[5]))]

    roll_mat = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rotations[0]), -np.sin(rotations[0])],
            [0, np.sin(rotations[0]), np.cos(rotations[0])],
        ]
    ).astype(np.float64)

    pitch_mat = np.array(
        [
            [np.cos(rotations[1]), 0, np.sin(rotations[1])],
            [0, 1, 0],
            [-np.sin(rotations[1]), 0, np.cos(rotations[1])],
        ]
    ).astype(np.float64)

    yaw_mat = np.array(
        [
            [np.cos(rotations[2]), -np.sin(rotations[2]), 0],
            [np.sin(rotations[2]), np.cos(rotations[2]), 0],
            [0, 0, 1],
        ]
    ).astype(np.float64)

    if method == 'vertical':
        points[:, :3] = np.matmul(points[:, :3], yaw_mat)

    elif method == 'all':
        points[:, :3] = np.matmul(np.matmul(np.matmul(points[:, :3], roll_mat), pitch_mat), yaw_mat)
    else:
        raise ValueError(f"Unsupported method : {method}")

    return points

def scale(points, scale_limits, anisotropic, rng):

    """Scale augmentation for pointcloud.
    If `scale_anisotropic` is True, each point is scaled differently.
    else, same scale from range ['min_s', 'max_s') is applied to each point.
    Args:
        points: Pointcloud to augment.
        scale_limits: List of two values [min, max], which are used as a range for uniform
                      random generation of scaling factor.
        anisotropic: Determines if each (True) point should be scaled individualy with given scale factor
                     or (False) all points are scaled with single random factor.
        rng: Random number generator initialized and shared in SegmantationAugmentor class.
    """

    if anisotropic:
        scale_factor = rng.uniform(scale_limits[0], scale_limits[1], points.shape[0])
    else:
        scale_factor = rng.uniform(scale_limits[0], scale_limits[1])

    return points[:,:3] * scale_factor


def random_noise_addition(points, deviation_limits, rng):

    """Random noise addition for pointcloud.
    Args:
        points: Pointcloud to augment.
        deviation_limits: List of two values [min, max], which are used as a range for uniform
                          random generation of std of  Gaussian distribution which generates
                          additional noise.
        rng: Random number generator initialized and shared in SegmantationAugmentor class.
    """

    random_noise_std_dev = rng.uniform(deviation_limits[0], deviation_limits[1])
    points[:,:3] = points[:,:3] + rng.normal(0, random_noise_std_dev, size=(np.shape(points)[0], 3))
    
    return points


    rotations = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-180, 180)]

    x = rotate_3d(x, rotations)
    x = random_scale_change(x, 0.8, 1.2)

    if np.random.uniform(0, 1) >= 0.5 and x.shape[0] > min_sample_points:
        x, y = subsample_point_cloud(x, y, np.random.uniform(0.01, 0.025), min_sample_points)

    if np.random.uniform(0, 1) >= 0.8 and x.shape[0] > min_sample_points:
        x, y = random_point_removal(x, y, min_sample_points)

    x = random_noise_addition(x)
    return x, y

def size_downsampler(points, max_points, rng):

    """Sampler which reduces point cloud size to given limit of points.
    Args:
        points: Pointcloud to augment.
        max_points: Maximal limit of points in point cloud.
        rng: Random number generator initialized and shared in SegmantationAugmentor class.
    """

    indices = np.arange(np.shape(points)[0])
    rng.shuffle(indices)

    return points[indices[:max_points],:]

    
def create_3D_rotations(axis, angle):
    """Create rotation matrices from a list of axes and angles. Code from
    wikipedia on quaternions.

    Args:
        axis: float32[N, 3]
        angle: float32[N,]

    Returns:
        float32[N, 3, 3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([
        t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20,
        t11 - t12, t19 + t20, t1 + t2 * t24
    ],
                 axis=1)

    return np.reshape(R, (-1, 3, 3))