from flask import Flask, jsonify, request
import io
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import json

app = Flask(__name__)
image_class_index = json.load(open("./static/imagenet_class_index.json"))
model = models.densenet121(pretrained=True)
model.eval()

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

@app.route('/predict', methods=["POST"])

def predict():
	if request.method == "POST":
		file = request.files["file"]
		img_bytes = file.read()
		class_id, class_name = get_prediction(image_bytes=img_bytes)
		return jsonify({"class_id": class_id, "class_name": class_name})

if __name__ == "__main__":
	app.run()

	

