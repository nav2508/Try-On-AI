from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import io
from PIL import Image
import base64
from datasets.deepfashion_datasets import DFVisualDataset
from apputils import dress_in_order, model, plot_img, inputs, dataroot, ds, categories

app = Flask(__name__)

index_val = sorted([len(inputs[i]) for i in inputs.keys()])[0]


@app.route("/get_pimg", methods=["POST"])
def get_pimg():
    data = request.json

    pose_category = data.get("category")
    garmet_category = data.get("gid")

    pid_index = data.get("pid_index")
    pose_index = data.get("g_index")

    # garment type (need to be added)
    g_type = data.get("g_type") # top or bottom
    enc = 5
    print(g_type)
    if g_type == "bottom":
        enc = 1

    if pose_category not in categories or garmet_category not in categories:
        return jsonify({"error": "Invalid category or GID selected"}), 400

    pid = (pose_category, pid_index, None)
    gids = [(garmet_category, pose_index, enc)]  
    
    pimg, gimgs, _, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2,1,5])
    
    img_pose = plot_img(pimg,pose=pose)
    garmet = plot_img(gimgs=gimgs)
    result = plot_img(gen_img=gen_img)

    return jsonify({"img_pose": img_pose, "garmet" : garmet, "generated_image": result})



@app.route("/")
def index():
    return render_template("index.html", categories=categories, index_val=index_val)

if __name__ == "__main__":
    app.run(debug=True,port = 5001)
