import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def ImagetoSketch(photo):
    # specify the path to image (Loading image image)
    # img = cv2.imread(r'C:\Users\rakib\OneDrive\GitHub\Image_to_sketch\src\images\demo_2.jpg')
    img = cv2.imread(photo)

    print('Original Dimensions : ', img.shape)
    # scale_percent = 80  # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)

    # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # print('Resized Dimensions : ', resized.shape)

    # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    window_name = 'Original Image'
    # DispLaying the original image
    cv2.imshow(window_name, img)

    # convert the image from one color space to another
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(grey_img)

    # image smoothing
    blur = cv2.GaussianBlur(invert, (111, 111), 0)
    invertedblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(grey_img, invertedblur, scale=256.0)

    # save the converted image to specified path
    cv2.imwrite(r"C:\Users\rakib\OneDrive\GitHub\Image_to_sketch\src\images\sketch_2.png", sketch)

    # Reading an image in default mode
    image = cv2.imread(r"C:\Users\rakib\OneDrive\GitHub\Image_to_sketch\src\images\sketch_2.png")

    # Window name in which image is displayed
    window_name = 'Sketch image'
    cv2. imshow(window_name, image) #waits for user to press any key #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.title('Original image', size=18)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Sketch', size=18)
    rgb_sketch = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_sketch)
    plt.axis('off')
    plt.show()


#ImagetoSketch(r'C:\Users\rakib\OneDrive\GitHub\Image_to_sketch\src\images\demo_2.png')
# python setup.py sdist bdist_wheel
# pip install -e .
# twine upload dist/*