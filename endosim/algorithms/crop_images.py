from PIL import Image
import os

if __name__ == '__main__':

    img_path= '/endosim/DenseDescriptorLearning-Pytorch/data/1/_start_002603_end_002984_stride_1000_segment_00/images'
    cropped_img_path = '/endosim/DenseDescriptorLearning-Pytorch/data/1/_start_002603_end_002984_stride_1000_segment_00/images_noncrop'

    # if not a directory, make directory
    if not os.path.isdir(cropped_img_path):
        os.mkdir(cropped_img_path)

    # resize and save images
    for image_path in os.listdir(img_path):
        # check if the image ends with png
        if (image_path.endswith(".jpg")):
            # image = Image.open('/Users/aure/Desktop/i4health/project/endoSim/endosim/DenseDescriptorLearning-Pytorch/data/1/_start_002603_end_002984_stride_1000_segment_00/images/00002603.jpg')
            image = Image.open(f'{img_path}/{image_path}')
            new_image = image.resize((320, 256))
            image.save(f'{cropped_img_path}/{image_path}')