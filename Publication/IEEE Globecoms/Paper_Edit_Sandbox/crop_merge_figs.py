from PIL import Image
import os

# Paths
mag_cdf_path = 'stage2_mag_sequence_cdf.png'
dual_cdf_path = 'stage3_dual_kalmannet.png'

# 1. Process Mag CDF (Fig 4)
img_mag = Image.open(mag_cdf_path)
# Crop out the white borders nicely. 
# We'll just auto-crop based on bounding box.
bg = Image.new(img_mag.mode, img_mag.size, img_mag.getpixel((0,0)))
diff = Image.composite(img_mag, bg, img_mag)
bbox = diff.getbbox()
if bbox:
    img_mag_cropped = img_mag.crop(bbox)
    img_mag_cropped.save('mag_cdf_cropped.png')

# 2. Process Dual KalmanNet CDF (Fig 5 side-by-side)
img_dual = Image.open(dual_cdf_path)
w, h = img_dual.size

# It's side-by-side, so let's cut it exactly in half width-wise
img_left = img_dual.crop((0, 0, w//2, h))
img_right = img_dual.crop((w//2, 0, w, h))

# Auto-crop the left image
bg = Image.new(img_left.mode, img_left.size, img_left.getpixel((0,0)))
diff = Image.composite(img_left, bg, img_left)
bbox = diff.getbbox()
if bbox:
    img_left = img_left.crop(bbox)
    img_left.save('kalman_full_wifi.png')

# Auto-crop the right image
bg = Image.new(img_right.mode, img_right.size, img_right.getpixel((w-1,0)))
diff = Image.composite(img_right, bg, img_right)
bbox = diff.getbbox()
if bbox:
    img_right = img_right.crop(bbox)
    img_right.save('kalman_degraded_wifi.png')

print("Successfully cropped and split CDF images!")
