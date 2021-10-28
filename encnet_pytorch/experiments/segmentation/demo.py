import torch
import encoding
# Get the model
model = encoding.models.get_model('encnet_resnet50d_citys', pretrained=False).cuda()
model.eval()

# print(model)

# Prepare the image
# url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
#       'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
# filename = 'example.jpg'
# img = encoding.utils.load_image(
#     encoding.utils.download(url, filename)).cuda().unsqueeze(0)
print("------1----------")
img = encoding.utils.load_image('ADE_val_00001142.jpg').cuda().unsqueeze(0)

# Make prediction
print("------2----------")
output = model.evaluate(img)
print("------3----------")
#predict = torch.max(output, 1)[1].cpu().numpy() + 1

# Get color pallete for visualization
#mask = encoding.utils.get_mask_pallete(predict, 'ade20k')
#mask.save('output.png')
