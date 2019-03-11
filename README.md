# MDBNet
This is the PyTorch version repo for Static Crowd Scene Analysis via Deep Network with Multi-branch Dilated Convolution Blocks

<h2>Prerequisites</h2>

Python: 2 or 3<br/>
PyTorch: 0.4.1
<h2>Ground Truth</h2>
Please follow the make_dataset.ipynb to generate the ground truth (In the python2). It shall take some time to generate the dynamic ground truth. Note you need to generate your own json file.
<h2>Train</h2>
Try python train.py train.json val.json 0 0 to start training process.(python3)
<h2>Test</h2>
Follow the RunTest.py.(python3)
