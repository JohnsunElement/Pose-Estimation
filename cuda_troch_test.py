import torch
from torchvision import models
import numpy as np


print(torch.cuda.is_available())

device = torch.device("cuda")
image = np.zeros((2,3,224, 224))
print( image.shape )
image_tensor = torch.from_numpy(image).type(torch.float).to(device)
print( image_tensor.shape )
model = models.resnet50(pretrained=True)
model = model.to(device)

out = model(image_tensor)
print(out)

# IPython.embed()
