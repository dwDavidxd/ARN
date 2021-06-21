from advertorch.test_utils import LeNet5
import torch
from advertorch_examples.utils import TRAINED_MODEL_PATH
import os
from torchvision import transforms
from utils.dataload import DatasetNPY_test
from torch.utils.data import DataLoader
from advertorch.utils import predict_from_logits
import pickle

def load_variavle(filename):
 f=open(filename,'rb')
 r=pickle.load(f)
 f.close()
 return r


torch.manual_seed(0)

input_dirs = './adv_example/processed/adv1'
label_dirs = './adv_example/test/label_true.pkl'

filename = "mnist_lenet5_clntrained.pt"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = LeNet5()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
model.to(device)
model.eval()

labels=load_variavle(label_dirs)

batch_size = 1000

# trans = transforms.Compose([transforms.Resize(size=[28, 28]),transforms.ToTensor()])
trans = transforms.ToTensor()

img_dataset = DatasetNPY_test(npy_dirs=input_dirs, transform=trans)

img_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

label_pred = []
label_true = []

for num in range(len(labels)):
    label_true.append(labels[num])

for img in img_loader:
    img = img.to(device)

    pred = predict_from_logits(model(img))

    for n in range(len(pred)):
        label_pred.append(pred.data[n].item())

acc1 = 0

for n in range(len(label_true)):
    if label_pred[n] == label_true[n]:
        acc1 += 1

print('Classification error rate: {:.3f}%'.format(100 * (1-acc1/len(label_true))))




