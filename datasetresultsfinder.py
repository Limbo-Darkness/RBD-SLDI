import result
from pathlib import Path

# gather images
imagedirectory = Path('image-loader')
data = []
loadcounter = 0
for images in imagedirectory.iterdir():
    imagename = images.name

    # filter out auxiliary ISIC dataset files
    if imagename.endswith('_superpixels.png'):
        continue
    elif imagename.endswith('_metadata.csv'):
        continue
    else:
        data.append(str(images))
        loadcounter+=1

print(f'{loadcounter} images loaded.')

# iterating through all 120 scenarios
# 6 different preprocessors
for i in range(0,6):
    if i == 0:
        pre = 'None'
    elif i == 1:
        pre = 'Gaussian Smoothing'
    elif i == 2:
        pre = 'Median Filtering'
    elif i == 3:
        pre = 'Bilateral Filtering'
    elif i == 4:
        pre = 'Histogram Equalization'
    elif i == 5:
        pre = 'CLAHE'
    # 5 different degradations
    for j in range(0,5):
        if j == 0:
            deg = 'None'
        elif j == 1:
            deg = 'Gaussian Noise'
        elif j == 2:
            deg = 'Salt and Pepper Noise'
        elif j == 3:
            deg = 'Blur'
        elif j == 4:
            deg = 'Reduce Illumination'
        #4 different edge detectors
        for k in range(0,4):
            if k == 0:
                edg = 'Sobel'
            elif k == 1:
                edg = 'Prewitt'
            elif k == 2:
                edg = 'LoG'
            elif k == 3:
                edg = 'Canny'
            # assemble code from iterative loop
            code = f'{i}{j}{k}'
            print(f'Code: {code}')
            # execute calculation step for all settings
            result.calculate(data,pre,deg,edg)