# Import libraries
import torchvision.transforms.functional as FF, numpy as np

class SquarePad:
  
  """
  
  This class gets an image and adds padding to make the image square.
  
  Parameter:
  
       image - an input image, array;
       
  Output:
  
       image - a square padded output image, array;
  
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
