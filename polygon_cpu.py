import cv2
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from argparse import ArgumentParser
# get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm

# os.listdir('../input/submissions')


ap = ArgumentParser()

ap.add_argument("--csv", dest = "csv", type = str, help = "csv file path")
ap.add_argument("--minsize", dest = "minsize", type = int, default = 10000)

args = ap.parse_args()

FILE = args.csv
MIN_SIZE = args.minsize
print(FILE)


# In[3]:


sub = pd.read_csv(FILE) # Source Mask


# We hide standard helper functions such as rle_decode/encode , etc.

# In[4]:


# helper functions
# credits: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools
class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
def rle_decode(mask_rle: str = '', shape = (1400, 2100)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')

def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(df, image_label, shape = (1400, 2100), cv_shape = (525, 350),debug=False):
    """
    Create mask based on df, image name and shape.
    """
    if debug:
        print(shape,cv_shape)
    df = df.set_index('Image_Label')
    encoded_mask = df.loc[image_label, 'EncodedPixels']
#     print('encode: ',encoded_mask[:10])
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded_mask is not np.nan:
        mask = rle_decode(encoded_mask,shape=shape) # original size
            
    return cv2.resize(mask, cv_shape)

min_size = [MIN_SIZE ,MIN_SIZE, MIN_SIZE, MIN_SIZE]

print("Min size")
print(min_size)
def post_process_minsize(mask, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(mask.shape)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions #, num


def draw_masks(img2,img_mask_list):
    
    img = img2.copy()
    for ii in range(4): # for each of the 4 masks
        color_mask = np.zeros(img2.shape)
        temp_mask = np.ones([img2.shape[0],img2.shape[1]])*127./255.
        temp_mask[img_mask_list[ii] == 0] = 0
        if ii < 3: # use different color for each mask
            color_mask[:,:,ii] = temp_mask
        else:
            color_mask[:,:,0],color_mask[:,:,1],color_mask[:,:,2] = temp_mask,temp_mask,temp_mask # broadcasting to 3 channels
    
        img += color_mask
        
    return img


# The following function is used to draw a convex-hull where you have four choices : 
# 
# * Convex Hull, (`mode = 'convex'`)
# * Simple xy-oriented rectangle, (`mode = 'rect'`)
# * Minimum-area rectangle and (`mode = 'min'`)
# * Approximate Polygon using [Douglas-Peucker algorithm](http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm) (`mode = 'approx'`)
# 
# Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

# In[6]:


def draw_convex_hull(mask, mode='convex'):
    
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        elif mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        elif mode=='approx':
            epsilon = 0.02*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255.


# ## Convex-Hull Shape
# To speed up the process a bit, I use @ryches dataset that already preprocess images to 350x525. Below is an example of using the function.

# In[7]:


mode='convex' # choose from 'rect', 'min', 'convex' and 'approx'
test_imgs_folder = 'test_images/'
NN=5
folder_images=test_imgs_folder

from PIL import Image
def open_image(path):
    img = Image.open(path)
    img =img.resize((525,350))
    return np.array(img).astype(np.float32)

images_list = ['e0cacb5.jpg', '05fcce5.jpg', 'e949e4b.jpg', 'ce8df6b.jpg', '41be432.jpg'] #os.listdir(folder_images)
current_batch = images_list[0: NN]
print(current_batch)

for i, image_name in enumerate(current_batch):
    path = os.path.join(folder_images, image_name)
    print(path)
    img = open_image(path)
    print(img.max(), img.min())
    img = img/255.
    
    img2 = img.mean(axis=-1)
    img2 = np.stack([img2,img2,img2],axis=-1)
         
    img_mask=[]
    img_mask_convex=[]
    img_mask_convex_minsize=[]
    
    for class_id in range(4):
        img_mask.append(make_mask(sub, image_name + '_' + class_names[class_id],shape=(350,525)))
        print(img_mask[class_id].min(), img_mask[class_id].max())
        img_mask_convex.append(draw_convex_hull(img_mask[class_id].astype(np.uint8), mode=mode))
        img_mask_convex[class_id][img2[:,:,0]<=2/255.] = 0
        img_mask_convex_minsize.append(post_process_minsize(img_mask_convex[class_id], min_size[class_id]) )
    
    img3 = draw_masks(img2,img_mask)
    img4 = draw_masks(img2,img_mask_convex_minsize)


# ## Approximated Polygon Shape (Douglas-Peucker algorithm) 

# In[ ]:


mode='approx' # choose from 'rect', 'min', 'convex' and 'approx'


test_imgs_folder = 'test_images/'
NN=5
folder_images=test_imgs_folder
images_list = ['e0cacb5.jpg', '05fcce5.jpg', 'e949e4b.jpg', 'ce8df6b.jpg', '41be432.jpg'] #os.listdir(folder_images)
current_batch = images_list[0: NN]
print(current_batch)

for i, image_name in enumerate(current_batch):
    path = os.path.join(folder_images, image_name)
    print(path)
    img = open_image(path) # use already-resized ryches' dataset
    print(img.max(), img.min())
    img = img/255.
    
    img2 = img.mean(axis=-1)
    img2 = np.stack([img2,img2,img2],axis=-1)
         
    img_mask=[]
    img_mask_convex=[]
    img_mask_convex_minsize=[]
    
    for class_id in range(4):
        img_mask.append(make_mask(sub, image_name + '_' + class_names[class_id],shape=(350,525)))
        print(img_mask[class_id].min(), img_mask[class_id].max())
        img_mask_convex.append(draw_convex_hull(img_mask[class_id].astype(np.uint8), mode=mode))
        img_mask_convex[class_id][img2[:,:,0]<=2/255.] = 0
        img_mask_convex_minsize.append(post_process_minsize(img_mask_convex[class_id], min_size[class_id]) )
    
    img3 = draw_masks(img2,img_mask)
    img4 = draw_masks(img2,img_mask_convex_minsize)

# After preparing everything, we convert all masks to the selected-mode rectangle below.


model_class_names=['Fish', 'Flower', 'Gravel', 'Sugar']

print(model_class_names)


mode='convex' # choose from 'rect', 'min', 'convex' and 'approx'

img_label_list = []
enc_pixels_list = []
test_imgs = os.listdir(folder_images)
for test_img_i, test_img in enumerate(tqdm(test_imgs)):
    for class_i, class_name in enumerate(model_class_names):
        
        path = os.path.join(folder_images, test_img)
        img = open_image(path)
        img = img/255.
        img2 = img.mean(axis=-1)
        
        img_label_list.append(f'{test_img}_{class_name}')
        
        mask = make_mask(sub, test_img + '_' + class_name,shape=(350,525))
        if True:
        #if class_name == 'Flower' or class_name =='Sugar': # you can decide to post-process for some certain classes 
            mask = draw_convex_hull(mask.astype(np.uint8), mode=mode)
        mask[img2<=2/255.] = 0
        mask = post_process_minsize(mask, min_size[class_i])
        
        if mask.sum() == 0:
            enc_pixels_list.append(np.nan)
        else:
            mask = np.where(mask > 0.5, 1.0, 0.0)
            enc_pixels_list.append(mask2rle(mask))
            

print("Creating dataframe")
submission_df = pd.DataFrame({'Image_Label': img_label_list, 'EncodedPixels': enc_pixels_list})

def bringOrder(df):
    return df.sort_values(by="Image_Label", ascending=True).reset_index().drop("index",axis=1)
# keep the original image order
submission_df = bringOrder(submission_df)

submission_df.to_csv('convex_%s'%(FILE), index=None)

print("File saved to:\t%s"%('convex_%s'%(FILE)))
sub_test=pd.read_csv('convex_%s'%(FILE))
sub_test.head(10)



