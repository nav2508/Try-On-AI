from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import base64
import os
import json
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
from flask import Flask, Response

# Import DIOR model
from models.dior_model import DIORModel
from datasets.deepfashion_datasets import DFVisualDataset

# Define dataset root and experiment details
dataroot = 'datas'
exp_name = 'DIORv1_64'  # DIOR_64
epoch = 'latest'
netG = 'diorv1'  # dior
ngf = 64

# Dummy argparse for model configuration
class Opt:
    def __init__(self):
        self.dataroot = dataroot
        self.isTrain = False
        self.phase = 'test'
        self.n_human_parts = 8
        self.n_kpts = 18
        self.style_nc = 64
        self.n_style_blocks = 4
        self.netG = netG
        self.netE = 'adgan'
        self.ngf = ngf
        self.norm_type = 'instance'
        self.relu_type = 'leakyrelu'
        self.init_type = 'orthogonal'
        self.init_gain = 0.02
        self.gpu_ids = []
        self.frozen_flownet = True
        self.random_rate = 1
        self.perturb = False
        self.warmup = False
        self.name = exp_name
        self.vgg_path = ''
        self.flownet_path = ''
        self.checkpoints_dir = 'checkpoints'
        self.frozen_enc = True
        self.load_iter = 0
        self.epoch = epoch
        self.verbose = False

# Initialize model
opt = Opt()
model = DIORModel(opt)
model.setup(opt)
model.eval()

# Load dataset
ds = DFVisualDataset(dataroot=dataroot, dim=(256, 176), n_human_part=8)
inputs = {attr: ds.get_attr_visual_input(attr) for attr in ds.attr_keys}

# Define categories
categories = ['plaid', 'plain', 'pattern', 'strip', 'print', 'collar', 'lace', 'gfla', 'flower', 'jacket', 'mixed']







def load_img(pid, ds):
    if isinstance(pid, int):  # If `pid` is an integer, convert it to a tuple with a default index
        pid = (str(pid), 0)  # Assuming default index is 0; adjust as needed

    if len(pid) < 10:  # Pre-selected models
        person = inputs[pid[0]]
        person = (i for i in person)
        pimg, parse, to_pose = person
        pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
    else:  # Load model from scratch
        person = ds.get_inputs_by_key(pid[0])
        person = (i for i in person)
        pimg, parse, to_pose = person

    return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()

def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None):
    if pose is not None:
        from utils.pose_utils import draw_pose_from_map
        kpt = draw_pose_from_map(pose.cpu().numpy().transpose(1, 2, 0), radius=6)[0]

    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]

    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2  # Denormalize
        out = np.transpose(out, [1, 2, 0])

        if pose is not None:
            out = np.concatenate((kpt, out), 1)
    else:
        out = kpt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.axis("off")
    ax.imshow(out)

    buf = BytesIO()
    fig.canvas.print_png(buf)  # Save as PNG
    plt.close(fig)

    # Convert to base64 string
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64  # Return as string



# define dressing-in-order function (the pipeline)
def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5,1,3,2]):
    PID = [0,4,6,7]
    GID = [2,5,1,3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)
        
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])

    # swap base garment if any
    gimgs = []
    for gid in gids:
        _,_,k = gid
        gimg, gparse, pose =  load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
        gsegs[gid[2]] = seg
        gimgs += [gimg * (gparse == gid[2])]

    # encode garment (overlay)
    garments = []
    over_gsegs = []
    oimgs = []
    for gid in ogids:
        oimg, oparse, pose = load_img(gid, ds)
        oimgs += [oimg * (oparse == gid[2])]
        seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
        over_gsegs += [seg]

    gsegs = [gsegs[i] for i in order] + over_gsegs
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    return pimg, gimgs, oimgs, gen_img[0], to_pose

