import SimpleITK
import nibabel as nib
from skimage.measure import label, regionprops
from skimage.restoration import denoise_bilateral
from scipy import ndimage as ndi
from skimage.filters import rank, gaussian
from skimage.morphology import watershed, disk
from skimage import exposure
import numpy as np


class BrainData:
    '''
    Class to store the data of MRI image and provide function to easily
    access slices of the MRI image
    '''

    TOP_PROF = 0
    FRONT_PROF = 1
    SIDE_PROF = 2

    def __init__(self, mha_location):

        is_mha = False

        if mha_location.endswith('.mha') :
            is_mha = True
            try:
                input_image = SimpleITK.ReadImage(mha_location)
            except Exception:
                raise Exception(f'Error opening file "{mha_location}"')
        

        elif mha_location.endswith('.nii.gz'):
            try:
                input_image = nib.load(mha_location)
            except:
                raise Exception(f'Error opening file "{mha_location}"')
        

        # do some checking here to make sure it is a 3d image

        # get the image data from mha
        if is_mha:
            self.data = normalize_255(SimpleITK.GetArrayFromImage(input_image))
        else:
            image = input_image.get_data()
            image = np.transpose(image, (2,1,0))
            self.data = normalize_255(image)

        self.dimensions = self.data.shape

    def get_slice(self, profile, index):

        '''
        Returns uint8 2d ndarray of a specific slice of the MRI image
        Returns None if error is encountered
        '''

        if index >= self.dimensions[profile]:
            return None

        if profile == self.TOP_PROF:
            return self.data[index, :, :]
        if profile == self.FRONT_PROF:
            return self.data[:, index, :]
        if profile == self.SIDE_PROF:
            return self.data[:, :, index]

    def get_dimensions(self):
        return self.dimensions
        
    
############################################
# HELPER FUNCTIONS
############################################

def isolate_brain(img_array):

    '''
    Lazy way of isolating the brain from the 2d mri image
    '''

    result = {'data': None, 'origin': (0, 0)}

    # binarize the image so we can properly separate the brain region.
    # hardcode the threshold for now. this is just 
    # a fast isolation method
    bin_data = img_array > 10

    # start labelling regions
    label_image = label(bin_data)

    # get the regions
    regions = regionprops(label_image)

    # if the number of regions is zero, there is no brain
    if(len(regions)) == 0:
        return result
    # the number of regions should be 1
    # if its greater than 1, find the largest one
    selected_region = 0
    max_area = 0

    if len(regions) > 1:
        for index, region in enumerate(regions):
            if region.area > max_area:
                selected_region = index
                max_area = region.area

    coords = regions[selected_region].bbox
    result['origin'] = (coords[2]-coords[0], coords[3] - coords[1])
    result['data'] = img_array[coords[0] : coords[2], coords[1] : coords[3]]

    return result


def segment(brain_img):

    '''
    Segmenting regions of the brain using watershed function
    '''
    # get the low gradient
    markers = rank.gradient(brain_img, disk(5)) < 20

    markers = ndi.label(markers)[0]

    # get the local gradient
    gradient = rank.gradient(brain_img, disk(1))

    # process the watershed
    labels = watershed(gradient, markers)

    return labels


def get_tumor_region(label, image):
    return None


def normalize_255(image):

    input_data = image
   

    min = np.amin(input_data)
    max = np.amax(input_data)
    
    normalized = (input_data-min) / (max-min) * 255

    normalized = np.clip(normalized, a_min=0, a_max=255)

    normalized = normalized.astype("uint8")

    #print(normalized)
    return normalized


def equalize(image, lower_bound=5, upper_bound=95):


    lb, ub = np.percentile(image, (lower_bound, upper_bound))
    img_rescale = exposure.rescale_intensity(image, in_range=(lb, ub))

    return img_rescale
 

def median(data):

    return rank.median(data, disk(1))

def bilateral(data, win_size=5, multichannel=False):

    return denoise_bilateral(data, win_size=win_size, multichannel=multichannel)

def watershed_segment(data):

    marker = rank.gradient(data, disk(1)) < 10
    marker = ndi.label(marker)[0]

    gradient = rank.gradient(data, disk(1))
    result = watershed(gradient, marker)

    return {
        'marker':marker,
        'gradient': gradient,
        'watershed': result
    } 
    

def detect_tumor(labeled_image, original_image, threshold=150):

    regions = regionprops(labeled_image, original_image)

    # area of the brain
    # mean intensity

    result = np.zeros(original_image.shape,dtype=np.uint8)
    area = 0

    for region in regions:
        if region.mean_intensity >= threshold:
            area += region.area
            min_row, min_col, max_row, max_col = region.bbox
            target_area = result[min_row:max_row, min_col:max_col] 
            result[min_row:max_row, min_col:max_col] = np.logical_or(target_area, region.image)

    return {
        'overlay': result,
        'original': original_image,
        'area': int(area)
    }


