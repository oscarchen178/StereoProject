import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def read_png(file_name, pil=False):
    img = Image.open(file_name)
    return np.array(img)

def readDispKITTI(file_name):
    """
    Read KITTI disparity map
    Returns:
        disp: Disparity map
        valid: Validity mask (disparity > 0)
    """
    disp = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid


# Currently not used

# def readFlowKITTI(file_name):
#     """
#     Read KITTI flow file
#     Returns:
#         flow: Optical flow map
#         valid: Validity mask
#     """
#     flow = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
#     flow = flow[:,:,::-1].astype(np.float32)
#     flow, valid = flow[:, :, :2], flow[:, :, 2]
#     flow = (flow - 2**15) / 64.0
#     return flow, valid

# def writeFlowKITTI(filename, uv):
#     """Write KITTI flow file"""
#     uv = 64.0 * uv + 2**15
#     valid = np.ones([uv.shape[0], uv.shape[1], 1])
#     uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
#     cv2.imwrite(filename, uv[..., ::-1])