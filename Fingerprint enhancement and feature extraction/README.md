# Project


To install dependencies:

    pip3 install -r requirements.txt

To run:

    python3 src/GUI.py

Link to data: https://drive.google.com/file/d/1GZqHFzTLDI-1rcOctHdf-c16VgagWocd/view


Weak light image enhancement using adaptive gamma transform:
For a general image with uneven low light, a gamma transform will not enhance the image evenly. As such an adaptive gamma transform approach was followed in this paper.

In the illumination-reflection model the brightness of any pixel in an image can be considered a product of its illumination component(I) and the reflective component(R).

F(x, y) = I(x, y)*R(x, y)

The illumination component spectrum is usually concentrated in a low frequency region which reflects the lighting environment during image capture. If this can be extracted from the image and made even then the effect of uneven lighting can be removed from the image.

Conversion from RGB to YUV space
Human eyes are more sensitive to luminescence than color. As such the paper aims to enhance luminescence to correct uneven illumination. This information cannot be captured effectively in RGB space and therefore we work in the YUV space where each color corresponds to two chrominance components(U, V) and one brightness component(Y). This luminescence component(Y) will be enhanced to fix uneven lighting while leaving the U and V components unchanged.



