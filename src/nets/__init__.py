from squeezeDet import SqueezeDet
from squeezeDetPlus import SqueezeDetPlus
from vgg16_convDet import VGG16ConvDet
from resnet50_convDet import ResNet50ConvDet

# # # feature map net name -> net class
def get_net(name):
  all_nets = {
    'squeezeDet'        : SqueezeDet,
    'squeezeDet+'       : SqueezeDetPlus,
    'vgg16+convDet'     : VGG16ConvDet,
    'resnet50+convDet'  : ResNet50ConvDet}
  if(name in all_nets):
    return all_nets[name]
  else:
    return all_nets["squeezeDet+"]