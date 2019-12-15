import io
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

import json

# resnet18 = models.resnet18(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
model = models.densenet121(pretrained=True)
model.eval()

image_class_index = json.load(open("./static/imagenet_class_index.json"))
print(len(image_class_index))

def transform_image(image_bytes):
	my_transform = transforms.Compose([transforms.Resize(255),
		transforms.CenterCrop(224), 
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
	image = Image.open(io.BytesIO(image_bytes))
	return my_transform(image).unsqueeze(0)

# with open("./test.jpg", 'rb') as f:
# 	image_bytes = f.read()
# 	tensor = transform_image(image_bytes)
# 	print(tensor)

def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model.forward(tensor)
	_, y_hat = outputs.max(1) # output: tensor([281])
	predcited_idx = str(y_hat.item())

	return image_class_index[predcited_idx]

with open("./test.jpg", 'rb') as f:
	image_bytes = f.read()
	y_hat = get_prediction(image_bytes)
	print(y_hat)