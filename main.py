import streamlit as st

import numpy as np
import io
from skimage.io import imread
from PIL import Image
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.feature import canny
from skimage import morphology
from skimage.morphology import reconstruction
from skimage.measure import label
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from matplotlib.pyplot import imshow

st.write("""
# My first leave count App!
Upload the image of your plant
""")

# im = imread("example_2.jpg")
uploaded_file = st.file_uploader('plant', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    byte_img = uploaded_file.getvalue()

    st.image(byte_img)

    jpeg_image = Image.open(io.BytesIO(byte_img))
    rgb_image = np.array(jpeg_image)[:, :, :3]

    # Convert to float: Important for subtraction later which won't work with uint8
    image = img_as_float(rgb_image)

    # Apply preprocessing operations
    image = gaussian_filter(image, 1)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    grayscale_dilated = rgb2gray(dilated)
    edges1 = canny(grayscale_dilated, sigma=1)
    edges_smooth = morphology.dilation(edges1, morphology.disk(5))
    edges_smooth = morphology.erosion(edges_smooth, morphology.disk(5))
    labels = label(edges_smooth, connectivity=2)
    num_leaves = len(np.unique(labels)) - 1

    st.write("*%d* - Number of leaves detected" % num_leaves)