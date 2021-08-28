import torch
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
import glob
import os
import warnings
warnings.filterwarnings("ignore")


def prediction(img_path):
    """
    ! FUNCTION FAILS IF THE IMAGE IS GREYSCALE
    """
    checkpoint = torch.load(path+r'/models/resnet.pt',
                            map_location=torch.device('cpu'))
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 128)
    model.load_state_dict(checkpoint)
    model.eval()

    classes = ['no mask', 'mask']
    transformer = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor()])
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()
    input = torch.autograd.Variable(image_tensor)
    output = model(input)
    index = output.data.numpy().argmax()
    pred = classes[index]
    return pred


def main():
    """
    TODO: serial the each new model with its performance on the test dataset
    TODO: Use OpenCV to find faces (masked and unmasked) & make bounding boxes
    TODO: Feed in the OpenCV Data to prediction()
    TODO: Display prediciton on the bounding boxes
    """
    from tqdm import tqdm
    path = os.getcwd()
    pred_path = path+'/data/test/'

    images_path = glob.glob(pred_path+'/*.jpg')
    pred_dict = {}
    for i in tqdm(images_path, desc='Loading Files'):
        pred_dict[i[i.rfind('/')+1:]] = prediction(i)

    for key, value in pred_dict.items():
        print(key, ':', value)


if __name__ == "__main__":
    main()
