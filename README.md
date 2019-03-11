# MDBNet
This is the PyTorch version repo for Static Crowd Scene Analysis via Deep Network with Multi-branch Dilated Convolution Blocks

<h2>Prerequisites</h2>

Python: 2 or 3
PyTorch: 0.4.1
<h2>Ground Truth</h2>
Please follow the make_dataset.ipynb to generate the ground truth. It shall take some time to generate the dynamic ground truth. Note you need to generate your own json file.
</h2>Training Process</h2>
Try python train.py train.json val.json 0 0 to start training process.
</h2>Testing</h2>
Follow the val.ipynb to try the validation. You can try to modify the notebook and see the output of each image.
