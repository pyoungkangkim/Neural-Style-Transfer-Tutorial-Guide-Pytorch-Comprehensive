import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch as t
import torch.nn.functional as F
from torchvision import transforms, models

tfms = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC), transforms.ToTensor()])
device = t.device('cuda') if t.cuda.is_available() else 'cpu'

style_img = tfms(Image.open('data/style.jpg'))[None].to(device, t.float)
content_img = tfms(Image.open('data/content.jpg'))[None].to(device, t.float)

def imshow(tensor, title=None):
    plt.imshow(transforms.ToPILImage()(tensor.cpu()[0]))
    plt.title(title)
    plt.show()

imshow(style_img, title='Style Image')
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):
    def __init__(self, content_target):
        super(ContentLoss, self).__init__()
        self.content_target = content_target

    def forward(self, x):
        self.loss = F.mse_loss(x, self.content_target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, style_target):
        super(StyleLoss, self).__init__()
        self.style_target = self.gram_matrix(style_target).detach()

    def gram_matrix(self, x):
        b, c, h, w = x.shape
        features = x.view(b * c, w * h)
        g_matrix = t.mm(features, features.t())
        return g_matrix.div(b * c * w * h)

    def forward(self, x):
        gram_prod = self.gram_matrix(x)
        self.loss = F.mse_loss(gram_prod, self.style_target)
        return x

class Vgg19Norm(nn.Module):
    def __init__(self):
        super(Vgg19Norm, self).__init__()
        self.mean = t.tensor([[[0.485]], [[0.456]], [[0.406]]]).to(device)
        self.std = t.tensor([[[0.229]], [[0.224]], [[0.225]]]).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
content_losses, style_losses = [], []
model = nn.Sequential(Vgg19Norm())

i = 0
for layer in vgg19.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
    if isinstance(layer, nn.ReLU):
        layer = nn.ReLU()

    name = f'{layer.__class__.__name__}_{i}'
    model.add_module(name, layer)

    content_layers = ['Conv2d_4']
    style_layers = ['Conv2d_1', 'Conv2d_2', 'Conv2d_3', 'Conv2d_4', 'Conv2d_5']
    if name in content_layers:
        content_target = model(content_img).detach()
        content_loss = ContentLoss(content_target)
        model.add_module(f'content_loss_{i}', content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        style_target = model(style_img).detach()
        style_loss = StyleLoss(style_target)
        model.add_module(f'style_loss_{i}', style_loss)
        style_losses.append(style_loss)

for i, e in reversed(list(enumerate(model))):
    if isinstance(e, (ContentLoss, StyleLoss)):
        break

model = model[:i + 1]
output_img = content_img.clone()
optimizer = t.optim.LBFGS([output_img.requires_grad_()])

i = 0
while i <= 300:
    def closure():
        global i
        output_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(output_img)
        style_loss = 0
        content_loss = 0

        for sl in style_losses:
            style_loss += sl.loss
        for cl in content_losses:
            content_loss += cl.loss

        style_lambda = 1000000
        content_lambda = 1
        style_loss *= style_lambda
        content_loss *= content_lambda

        loss = style_loss + content_loss
        loss.backward()

        i += 1
        if i % 50 == 0:
            print('iteration: ', i)
            print(f'style loss : {style_loss.item()}\tcontent loss: {content_loss.item()}\n')

        return loss

    optimizer.step(closure)
output_img.data.clamp_(0, 1)

imshow(output_img, title='Output Image')
plt.show()
