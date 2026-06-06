import numpy as np
import torch
from easydict import EasyDict as edict
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing
from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


def load_configs_model(model_name='darknet', configs=None):
    if configs is None:
        configs = edict()

    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))

    if model_name == 'darknet':
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.1
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        configs.batch_size = 4
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False
    else:
        raise ValueError("Invalid model name")

    configs.no_cuda = True
    configs.gpu_idx = 0
    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')
    return configs


def load_configs(model_name='fpn_resnet', configs=None):
    if configs is None:
        configs = edict()

    configs.lim_x = [0, 50]
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0]
    configs.bev_width = 608
    configs.bev_height = 608
    configs.down_ratio = 4
    configs.num_classes = 3

    configs = load_configs_model(model_name, configs)
    configs.output_width = 608
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
    return configs


def create_model(configs):
    assert os.path.isfile(configs.pretrained_filename), \
        f"No model file at {configs.pretrained_filename}"

    if configs.arch == 'darknet' and configs.cfgfile is not None:
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)

    elif 'fpn_resnet' in configs.arch:
        # Output heads match the pretrained ResNet-18 FPN weights
        heads = {
            'hm_cen': 3,      # center heatmap for 3 classes
            'cen_offset': 2,  # x/y center offsets
            'direction': 2,   # sin/cos orientation encoding
            'z_coor': 1,      # z-coordinate
            'dim': 3          # width, length, height
        }
        model = fpn_resnet.get_pose_net(
            num_layers=18, heads=heads, head_conv=64, imagenet_pretrained=True
        )
    else:
        raise ValueError("Undefined model backbone")

    model.load_state_dict(torch.load(
        configs.pretrained_filename, map_location='cpu', weights_only=True
    ))
    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')
    model = model.to(device=configs.device)
    model.eval()
    return model


def detect_objects(input_bev_maps, model, configs):
    with torch.no_grad():
        outputs = model(input_bev_maps)

        if 'darknet' in configs.arch:
            output_post = post_processing_v2(
                outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh
            )
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                for obj in output_post[sample_i]:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])

        elif 'fpn_resnet' in configs.arch:
            decoded = decode(
                outputs['hm_cen'], outputs['cen_offset'],
                outputs['direction'], outputs['z_coor'], outputs['dim'],
                K=100
            )
            detections = post_processing(decoded.cpu().numpy(), configs)

    objects = []
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    if isinstance(detections, list) and len(detections) > 0:
        if isinstance(detections[0], dict):
            items_to_loop = detections[0].items()
        else:
            items_to_loop = enumerate(detections)

        for class_id, class_array in items_to_loop:
            if class_array is None or len(class_array) == 0:
                continue
            class_array = np.atleast_2d(class_array)
            for det in class_array:
                x = det[2] * bev_discretization + configs.lim_x[0]
                y = det[1] * bev_discretization + configs.lim_y[0]
                w = det[5] * bev_discretization
                l = det[6] * bev_discretization
                final_class = int(class_id) if isinstance(class_id, (int, float)) else 1
                objects.append([final_class, x, y, det[3], det[4], w, l, det[7]])

    return objects