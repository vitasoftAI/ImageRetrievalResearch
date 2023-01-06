import torchvision.transforms.functional as FF
import numpy as np

class SquarePad:
  
  """
  
  
  
  def __call__(self, image):
      w, h = image.size
      max_wh = np.max([w, h])
      hp = int((max_wh - w)/2)
      hp_rem = (max_wh - w)%2
      vp = int((max_wh - h)/2)
      vp_rem = (max_wh - h)%2
      padding = (hp, vp, hp+hp_rem, vp+vp_rem)
      return FF.pad(image, padding, 255, 'constant')
