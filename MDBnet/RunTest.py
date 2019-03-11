import glob
from image import *
from model import MDBNet
from torch.autograd import Variable
import torch
import math

from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

root = './data/Shanghai/'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = MDBNet()
model = model.cuda()
model.eval()

checkpoint = torch.load('A_lr_5model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

mae = 0
mse = 0
for i in range(len(img_paths)):
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    t = output.detach().cpu().sum().numpy()-np.sum(groundtruth)
    mae += abs(t)
    mse += t*t
print (mae/len(img_paths))
print(math.sqrt(mse/len(img_paths)))
