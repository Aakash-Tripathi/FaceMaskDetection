import torch
from torchvision import *
import pathlib
import glob
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn


path = 'D:/Projects/FaceMaskDetection/'
#pred_path = 'D:/Projects/FaceMaskDetection/data/pred-from-set/'
pred_path = 'D:/Projects/FaceMaskDetection/data/pred-new/'
train_path = path+'FMD/'
root = pathlib.Path(train_path)
classes = ['no mask', 'mask']

checkpoint = torch.load(path+'models/resnet.pt')
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 128)
model.load_state_dict(checkpoint)
model.eval()


def prediction(img_path):
    transformer = transforms.Compose([transforms.Resize((150, 150)),
                                      transforms.ToTensor()])
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()
    input = Variable(image_tensor)
    output = model(input)
    index = output.data.numpy().argmax()
    pred = classes[index]
    return pred


images_path = glob.glob(pred_path+'/*.jpg')
pred_dict = {}
for i in images_path:
    pred_dict[i[i.rfind('/')+1:]] = prediction(i)

for key, value in pred_dict.items():
    print(key, ':', value)
