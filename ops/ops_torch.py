import torch 
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple
from pytorch3d import _C
import numpy as np
import random
import copy

# BBOX MANIPULATIONS
#######################################################################

def bbox2rotated_corners2D(bbxs):

    # Define axis aligned vertices
    p_0 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 

    p_1 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 
                    
    p_2 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 
                    
    p_3 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 

    p_4 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_5 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_6 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_7 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    vertices = torch.cat([p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7], axis = 1).reshape(-1,8,3)

    #Define rotation
    x1 = torch.cat([torch.ones((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)], 
                    axis = 1)

    x2 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.cos(bbxs[...,6].unsqueeze(-1)), 
                    -torch.sin(bbxs[...,6].unsqueeze(-1))],
                    axis = 1)

    x3 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.sin(bbxs[...,6].unsqueeze(-1)), 
                    torch.cos(bbxs[...,6].unsqueeze(-1))],
                    axis = 1)

    rot_x = torch.cat([x1,x2,x3], axis = 1).reshape(-1,3,3)

    y1 = torch.cat([torch.cos(bbxs[...,7].unsqueeze(-1)), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.sin(bbxs[...,7].unsqueeze(-1))],
                    axis = 1)

    y2 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.ones((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    y3 = torch.cat([-torch.sin(bbxs[...,7].unsqueeze(-1)), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.cos(bbxs[...,7].unsqueeze(-1))],
                    axis = 1)

    rot_y = torch.cat([y1,y2,y3], axis = 1).reshape(-1,3,3)

    z1 = torch.cat([torch.cos(bbxs[...,8].unsqueeze(-1)), 
                    -torch.sin(bbxs[...,8].unsqueeze(-1)),
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    z2 = torch.cat([torch.sin(bbxs[...,8].unsqueeze(-1)), 
                    torch.cos(bbxs[...,8].unsqueeze(-1)), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    z3 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.ones((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    rot_z = torch.cat([z1,z2,z3], axis = 1).reshape(-1,3,3)

    rot = torch.matmul(torch.matmul(rot_z[:,...],rot_y[:,...]), rot_x[:,...])

    # Define center
    center = torch.cat([bbxs[...,0].unsqueeze(-1),bbxs[...,1].unsqueeze(-1),bbxs[...,2].unsqueeze(-1)], axis = -1)
    center = center.unsqueeze(1).repeat(1, 8, 1).reshape(-1,8,3)
    
    centered_verts = vertices - center
    rot_vertices = torch.matmul(centered_verts[:,...], rot.transpose(-2, -1)[:,...]) + center

    coord_min, _ = rot_vertices.min(dim=-2)
    coord_max, _ = rot_vertices.max(dim=-2)  

    return torch.cat([coord_min[:,:2],coord_max[:,:2]], axis=1)


def bbox2corners2D(bbxs):

     # Define axis aligned vertices
    p_0 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 

    p_1 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 
                    
    p_2 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 
                    
    p_3 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 

    p_4 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_5 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_6 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_7 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    vertices = torch.cat([p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7], axis = 1).reshape(-1,8,3)

    coord_min, _ = vertices.min(dim=-2)
    coord_max, _ = vertices.max(dim=-2)  

    return torch.cat([coord_min[:,:2],coord_max[:,:2], bbxs[...,8].unsqueeze(-1)], axis=1)


def bbox2corners3D(bbxs):

    # Define axis aligned vertices
    p_0 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 

    p_1 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 
                    
    p_2 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 
                    
    p_3 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]).unsqueeze(-1)), axis=1) 

    p_4 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_5 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]-bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_6 = torch.cat(((bbxs[...,0]+bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    p_7 = torch.cat(((bbxs[...,0]-bbxs[...,3]*0.5).unsqueeze(-1), 
                    (bbxs[...,1]+bbxs[...,4]*0.5).unsqueeze(-1), 
                    (bbxs[...,2]+bbxs[...,5]).unsqueeze(-1)), axis=1) 

    vertices = torch.cat([p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7], axis = 1).reshape(-1,8,3)
    
    # Define center
    center = torch.cat([bbxs[...,0].unsqueeze(-1),bbxs[...,1].unsqueeze(-1),bbxs[...,2].unsqueeze(-1)], axis = -1)
    center = center.unsqueeze(1).repeat(1, 8, 1).reshape(-1,8,3)

    #Define rotation
    x1 = torch.cat([torch.ones((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)], 
                    axis = 1)

    x2 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.cos(bbxs[...,6].unsqueeze(-1)), 
                    -torch.sin(bbxs[...,6].unsqueeze(-1))],
                    axis = 1)

    x3 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.sin(bbxs[...,6].unsqueeze(-1)), 
                    torch.cos(bbxs[...,6].unsqueeze(-1))],
                    axis = 1)

    rot_x = torch.cat([x1,x2,x3], axis = 1).reshape(-1,3,3)

    y1 = torch.cat([torch.cos(bbxs[...,7].unsqueeze(-1)), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.sin(bbxs[...,7].unsqueeze(-1))],
                    axis = 1)

    y2 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.ones((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    y3 = torch.cat([-torch.sin(bbxs[...,7].unsqueeze(-1)), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.cos(bbxs[...,7].unsqueeze(-1))],
                    axis = 1)

    rot_y = torch.cat([y1,y2,y3], axis = 1).reshape(-1,3,3)

    z1 = torch.cat([torch.cos(bbxs[...,8].unsqueeze(-1)), 
                    -torch.sin(bbxs[...,8].unsqueeze(-1)),
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    z2 = torch.cat([torch.sin(bbxs[...,8].unsqueeze(-1)), 
                    torch.cos(bbxs[...,8].unsqueeze(-1)), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    z3 = torch.cat([torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.zeros((bbxs.shape[0],1), device=bbxs.device), 
                    torch.ones((bbxs.shape[0],1), device=bbxs.device)],
                    axis = 1)

    rot_z = torch.cat([z1,z2,z3], axis = 1).reshape(-1,3,3)

    rot = torch.matmul(torch.matmul(rot_z[:,...],rot_y[:,...]), rot_x[:,...])
    centered_verts = vertices - center

    return torch.matmul(centered_verts[:,...], rot.transpose(-2, -1)[:,...]) + center


#https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html

def corner2faces3d(corners):
    """Convert 3d box corners from corner function above to surfaces that normal
    vectors all direct to internal.
    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).
    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """

    _box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],]

    faces = torch.tensor(_box_planes, dtype=torch.int64, device=corners.device)
    surfaces = corners.index_select(index=faces.view(-1), dim=1)

    return surfaces.reshape(-1,6,4,3)

def get_boxes_normals(boxes):

    corners = bbox2corners3D(boxes)
    norm_x = (corners[:,1,:] - corners[:,0,:]) / boxes[...,3].unsqueeze(-1)
    norm_y =(corners[:,3,:] - corners[:,0,:]) / boxes[...,4].unsqueeze(-1)
    norm_z = (corners[:,4,:] - corners[:,0,:]) / boxes[...,5].unsqueeze(-1)
    normal_vecs = torch.cat([norm_x, norm_y, norm_z], axis = -1)

    return torch.cat([boxes[..., :6], normal_vecs], axis = -1)

def points_in_boxes_old(points, boxes):

    inlier_indices = []
    bbxs_norms = get_boxes_normals(boxes)
    
    mask = np.zeros(shape = (points.shape[0],))

    for bbx in bbxs_norms:
        
        dir_vec = points[:,:3] - bbx[:3]

        res1 = torch.where( (torch.absolute(torch.matmul(dir_vec, bbx[6:9].unsqueeze(-1))) * 2) > bbx[3] )[0].detach().numpy()
        res2 = torch.where( (torch.absolute(torch.matmul(dir_vec, bbx[9:12].unsqueeze(-1))) * 2) > bbx[4] )[0].detach().numpy()
        res3 = torch.where( (torch.absolute(torch.matmul(dir_vec, bbx[12:].unsqueeze(-1))) * 2) > bbx[5] )[0].detach().numpy()

        inlier_indices.append(list(set(range(0, points.shape[0])) - set().union(res1, res2, res3)))

    mask[[item for sublist in inlier_indices for item in sublist]] = 1

    return mask.astype('bool')

def points_in_boxes_vectorized(points, boxes):

    normals = get_boxes_normals(boxes)
    normals[:,2] = normals[:,2] + normals[:,5]/2
    
    points = points[:,:3].unsqueeze(0).repeat(normals.size(0),1,1)

    center = normals[:,:3].repeat_interleave(points.size(1),dim=0).reshape(points.size(0),points.size(1),3)
    dir_vec = points-center

    res1 = torch.absolute(torch.bmm(dir_vec,normals[:,6:9].unsqueeze(-1)).squeeze(-1))*2 < normals[:,3].unsqueeze(-1)
    res2 = torch.absolute(torch.bmm(dir_vec,normals[:,9:12].unsqueeze(-1)).squeeze(-1))*2 < normals[:,4].unsqueeze(-1)
    res3 = torch.absolute(torch.bmm(dir_vec,normals[:,12:].unsqueeze(-1)).squeeze(-1))*2 < normals[:,5].unsqueeze(-1)
    
    return (res1*res2*res3).T

def points_in_boxes(points, boxes):

    normals = get_boxes_normals(boxes)
    normals[:,2] = normals[:,2] + normals[:,5]/2

    mask = torch.zeros(size = (points.shape[0],boxes.shape[0]))

    for i, bbx in enumerate(normals):
        
        dir_vec = points[:,:3] - bbx[:3]

        dir_vec = dir_vec.unsqueeze(0)
        bbx = bbx.unsqueeze(0)

        res1 = torch.absolute(torch.bmm(dir_vec,bbx[:,6:9].unsqueeze(-1)).squeeze(-1))*2 < bbx[:,3].unsqueeze(-1)
        res2 = torch.absolute(torch.bmm(dir_vec,bbx[:,9:12].unsqueeze(-1)).squeeze(-1))*2 < bbx[:,4].unsqueeze(-1)
        res3 = torch.absolute(torch.bmm(dir_vec,bbx[:,12:].unsqueeze(-1)).squeeze(-1))*2 < bbx[:,5].unsqueeze(-1)

        mask[:,i] = (res1*res2*res3).squeeze(0)

    return mask.T.any(axis=0).detach().cpu().numpy().tolist()


def get_semantic_label(points, boxes, ground_level, effective_trunk):

    normals = get_boxes_normals(boxes)
    normals[:,2] = normals[:,2] + normals[:,5]/2

    mask = torch.zeros(size = (points.shape[0],boxes.shape[0]))
    label = torch.zeros(size = (points.shape[0],))

    for i, bbx in enumerate(normals):
        
        box_label = torch.zeros(size = (points.shape[0],1))
        dir_vec = points[:,:3] - bbx[:3]
        dir_vec = dir_vec.unsqueeze(0)
        bbx = bbx.unsqueeze(0)

        res1 = torch.absolute(torch.bmm(dir_vec,bbx[:,6:9].unsqueeze(-1)).squeeze(-1))*2 < bbx[:,3].unsqueeze(-1)
        res2 = torch.absolute(torch.bmm(dir_vec,bbx[:,9:12].unsqueeze(-1)).squeeze(-1))*2 < bbx[:,4].unsqueeze(-1)
        res3 = torch.absolute(torch.bmm(dir_vec,bbx[:,12:].unsqueeze(-1)).squeeze(-1))*2 < bbx[:,5].unsqueeze(-1)

        bbx = bbx.squeeze(0)
        inbox_idx = (res1*res2*res3).squeeze(0)
        ground_idx = points[:,2] < bbx[2] - 0.5*bbx[5] + ground_level
        noisy_idx = points[:,2] > bbx[2] - 0.5*bbx[5] + effective_trunk* bbx[5]

        box_label[inbox_idx] = 1
        box_label[ground_idx * inbox_idx] = 0
        box_label[inbox_idx * noisy_idx] = -1
        mask[:,i] = box_label.squeeze(-1)
    
    label[(mask == 1).any(axis=1)] = 1
    label[(mask == -1).any(axis=1)] = -1

    return label

def remove_points_in_boxes(points, normals):
    """Remove the points in the sampled bounding boxes.
    Args:
        points (np.ndarray): Input point cloud array.
        boxes (np.ndarray): Sampled ground truth boxes.
    Returns:
        np.ndarray: Points with those in the boxes removed.
    """
    masks = points_in_boxes(points[:,:3], normals)
    points = points[torch.logical_not(masks.any(axis=1))]

    return points

def box_collision_test(boxes, qboxes):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
    """

    boxes = bbox2rotated_corners2D(boxes)
    qboxes = bbox2rotated_corners2D(qboxes)

    coll_mat = bbox_iou2D(boxes, qboxes)

    coll_mat[coll_mat != 0] = 1

    return coll_mat > 0

def torch_cov(input_vec:torch.tensor):

    x = input_vec - torch.mean(input_vec,axis=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

def get_min_bbox(points):

    """Return minimum bounding box encapsulating points.
    Args:
        points (np.ndarray): Input point cloud array.
    Returns:
        np.ndarray: 3D bounding box (x, y, z, w, h, l, yaw).
    """

    #z-dimension, vertical boxes are assumed
    h_min = torch.min(points[:,2])
    h_max = torch.max(points[:, 2])

    # just xy hyperplane -> rotation around z-axis
    points = points[:, :2]
    #covariance matrix
    cov_points = torch_cov(points)
    #Eigenvalues and eigenvectors
    val, vect = torch.eig(cov_points, True)
    tvect = vect.T
    #Rotate points to principal component coordinate system (defined by eigenvectors)
    points_rot = torch.matmul(points, torch.linalg.inv(tvect))

    # Mininal and maximal value in each PC
    min_a = torch.min(points_rot, axis=0)[0]
    max_a = torch.max(points_rot, axis=0)[0]

    # Size of point cloud in each PC
    diff = max_a - min_a
    center = min_a + diff * 0.5

    # Rotate center back to original coordinate system
    center = torch.matmul(center, tvect)
    center = torch.Tensor([center[0], center[1], (h_min + h_max) * 0.5])

    # Size of box
    width = diff[0]
    length = diff[1]
    height = h_max - h_min
    # Yaw rotation of box
    yaw = torch.atan(tvect[0, 1] / tvect[0, 0])
    
    return torch.Tensor([center[0], center[1], center[2], width, length, height, yaw])

def random_sample(files, num):
    if len(files) <= num:
        return files

    return random.sample(files, num)

def sample_class(num, gt_boxes, db_boxes):
    
    if num == 0:
        return []

    sampled = random_sample(db_boxes, num)
    
    sampled = copy.deepcopy(sampled)

    num_gt = len(gt_boxes)
    num_sampled = len(sampled)

    boxes = gt_boxes.copy()
    
    for box in sampled: boxes.append(box['bbox'])
    bboxes = torch.Tensor(boxes)

    coll_mat = box_collision_test(bboxes, bboxes)
    diag = torch.arange(len(bboxes))
    coll_mat[diag, diag] = False

    valid_samples = []
    for i in range(num_gt, num_gt + num_sampled):
        if coll_mat[i].any():
            coll_mat[i] = False
            coll_mat[:, i] = False
        else:
            valid_samples.append(sampled[i - num_gt])

    return valid_samples

def surface_normals(corners):

    """Compute normal vectors for polygon surfaces.
    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
    Returns:
        tuple: normal vector and its direction.
    """
    
    faces = corner2faces3d(corners)
    surf_vec = faces[:, :, :2, :] - faces[:, :, 1:3, :]
    normal_vec = torch.cross(surf_vec[:, :, 0, :],surf_vec[:, :, 1, :])
    direction = torch.einsum('aij, aij->ai', normal_vec, faces[:, :, 0, :])

    return normal_vec, -direction

def filter_by_min_points(bboxes, min_points_dict):
    """Filter ground truths by number of points in the bbox."""
    filtered_boxes = []

    for box in bboxes:
        if box['label'] in min_points_dict.keys():
            if box['points_inside_box'].shape[0] > min_points_dict[box['label']]:
                filtered_boxes.append(box)
        else:
            filtered_boxes.append(box)

    return filtered_boxes

# BBOX Intersections
#######################################################################

def bbox_iou2D(bboxes1, bboxes2, mode='iou',  eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)


    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])
    lt = torch.max(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
    overlap = wh[..., 0] * wh[..., 1]

    if mode in ['iou', 'giou']:
        union = area1[..., None] + area2[..., None, :] - overlap
    else:
        union = area1[..., None]

    if mode == 'giou':
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if mode in ['iou', 'iof']:
        return ious
        
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    
    """ 
    _box_planes define the 4-connectivity of the 8 box corners.
    _box_planes gives the quad faces of the 3D box
    """

    _box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
                  ]
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)

    coplanarity_tag = (mat1.bmm(mat2).abs() < eps).squeeze(1).squeeze(1)

    if coplanarity_tag.sum().item() != boxes.shape[0]:
        msg = "Plane vertices are not coplanar. This applies for bboxes in positions: {}".format(
               torch.arange(0, boxes.shape[0])[~coplanarity_tag])
        raise ValueError(msg)

    return


def check_nonzero(boxes: torch.Tensor, eps: float = 1e-4) -> None:

    """
    Checks that the sides of the box have a non zero area
    _box_triangles define the 3-connectivity of the 8 box corners.
    _box_triangles gives the triangle faces of the 3D box
    """
    _box_triangles = [
        [0, 1, 2],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [1, 5, 6],
        [1, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [3, 2, 6],
        [3, 6, 7],
        [0, 1, 5],
        [0, 4, 5],
    ]

    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    face_tag = torch.logical_not((face_areas < eps).any(dim=1))

    if face_tag.sum().item() != boxes.shape[0]:
        msg = "Planes have zero areas. This applies for bboxes in positions: {}".format(
                torch.arange(0, boxes.shape[0])[~face_tag])
        raise ValueError(msg)

    return

class _box3d_overlap(Function):
    """
    Torch autograd Function wrapper for box3d_overlap C++/CUDA implementations.
    Backward is not supported.
    """

    @staticmethod
    def forward(ctx, boxes1, boxes2):
        """
        Arguments defintions the same as in the box3d_overlap function
        """
        vol, iou = _C.iou_box3d(boxes1, boxes2)
        return vol, iou

    @staticmethod
    def backward(ctx, grad_vol, grad_iou):
        raise ValueError("box3d_overlap backward is not supported")


def box3d_overlap(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the intersection of 3D boxes1 and boxes2.
    Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes1 and boxes1),
    containing the 8 corners of the boxes, as follows:
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit boundingTrue
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    Args:
        boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        vol: (N, M) tensor of the volume of the intersecting convex shapes
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")

    check_coplanar(boxes1, eps)
    check_coplanar(boxes2, eps)
    check_nonzero(boxes1, eps)
    check_nonzero(boxes2, eps)

    vol, iou = _box3d_overlap.apply(boxes1, boxes2)

    return iou

# POINT CLOUD AUGMENTATIOS
#######################################################################
############################################################################
#Section of operations without random effect

def rotate_3d(points: torch.Tensor, rotations: torch.Tensor):

    rotations[0] = torch.deg2rad(rotations[0])
    rotations[1] = torch.deg2rad(rotations[1])
    rotations[2] = torch.deg2rad(rotations[2])

    roll_mat = torch.Tensor(
        [
            [1, 0, 0],
            [0, torch.cos(rotations[0]), -torch.sin(rotations[0])],
            [0, torch.sin(rotations[0]), torch.cos(rotations[0])],
        ]
    )

    pitch_mat = torch.Tensor(
        [
            [torch.cos(rotations[1]), 0, torch.sin(rotations[1])],
            [0, 1, 0],
            [-torch.sin(rotations[1]), 0, torch.cos(rotations[1])],
        ]
    )

    yaw_mat = torch.Tensor(
        [
            [torch.cos(rotations[2]), -torch.sin(rotations[2]), 0],
            [torch.sin(rotations[2]), torch.cos(rotations[2]), 0],
            [0, 0, 1],
        ]
    )

    points[:,:3] = torch.matmul(torch.matmul(torch.matmul(points[:, :3], roll_mat), pitch_mat), yaw_mat) 

    return points