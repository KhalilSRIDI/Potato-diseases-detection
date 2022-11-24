import cv2
import numpy as np
import os
import sys
import re


n = len(sys.argv)
for i in range(1, n):
    if n == 1 or sys.argv[i] == '-h':
        print('Arg1 : segment one image or whole dataset (-f or -d)\n')
        print('Arg2 : segmentation type: normal or HSV (-n or -hsv)\n')
        print('Arg3 : path (must be between double quotes "" )\n')
        exit()
    else:
        nature=sys.argv[1]
        type=sys.argv[2]
        path=sys.argv[3]
        continue
   
def segment_image(file):
  """
  segment a given image,remove it's backgreound and save the output result

  Keyword arguments:
  file -- the path of the file used
  Return: none
  """

  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  lower_green = np.array([25, 0, 20])
  upper_green = np.array([100, 255, 255])
  mask = cv2.inRange(hsv_img, lower_green, upper_green)
  result = cv2.bitwise_and(img, img, mask=mask)
  lower_brown = np.array([10, 0, 10])
  upper_brown = np.array([30, 255, 255])
  disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
  disease_result = cv2.bitwise_and(img, img, mask=disease_mask)
  final_mask = mask + disease_mask
  final_result = cv2.bitwise_and(img, img, mask=final_mask)
  new_path = str(path.rsplit("\\",3)[0])
  dir_path=os.path.join(new_path,"Segmented")
  output_path = new_path+os.path.basename(os.path.dirname(file))+"\\"+os.path.basename(file)
  cv2.imwrite(output_path, final_result)


def segment_image_HSV(file):
    """
        segment a given image into HSV color space and save the output result

    Keyword arguments:
    file -- the path of the file used
    Return: none
    """
    img = cv2.imread(file)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_path = str(path.rsplit("\\",3)[0])
    dir_path=os.path.join(new_path,"Segmented_HSV")
    output_path = new_path+os.path.basename(os.path.dirname(file))+"\\"+os.path.basename(file)
    cv2.imwrite(output_path,hsv_img)



def convert_dataset(rootdir,type):
    """
    convert the entire dataset into segmented images
    
    Keyword arguments:
    rootdir -- give the dataset root directory
    type -- give the type of dataset used (Color,Segmented)
    Return: return a processed dataset
    """  
    count=0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fpath=str(subdir)+'\\'+str(file)
            if type == 'Color':
                segment_image(fpath)
                count+=1
            else:
                segment_image_HSV(fpath)
                count+=1


    print('segmented '+str(count)+' images correctly')  
    
if nature=='-f':
    print('done')
    if type =='-n':
        segment_image(path)
    else:
        segment_image_HSV(path)
else:
    dirpath=path+'\\'
    if type =='-n':   
        convert_dataset(dirpath,'Color')
    else:
        convert_dataset(dirpath,'Segmented')
