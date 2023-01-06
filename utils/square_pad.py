import torchvision.transforms.functional as FF
import numpy as np

class SquarePad:
  
  """
  
  Gets an image and adds padding to make the image square.
  
  Argument:
  image - input image
  
  """

  def __call__(self, image):
    
      # Get width and height of the image
      w, h = image.size
      
      # Get max values of the width and height
      max_wh = np.max([w, h])
      
      # Create padding
      hp = int((max_wh - w)/2)
      hp_rem = (max_wh - w)%2
      vp = int((max_wh - h)/2)
      vp_rem = (max_wh - h)%2
      padding = (hp, vp, hp+hp_rem, vp+vp_rem)
      
      # Apply padding and return the padded image
      return FF.pad(image, padding, 255, 'constant')
