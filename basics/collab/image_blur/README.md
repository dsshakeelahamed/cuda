## Black and white blur

`python load_image.py` \
`!nvcc -arch=sm_75 -o blur blur.cu` \
`!./blur input.bin output.bin <width> <height> <radius>` \


## Color blur

`python load_color_image.py` \
`!nvcc -arch=sm_75 -o blur blur.cu` \
`!./blur input.bin output.bin <width> <height> <channels> <radius>` \