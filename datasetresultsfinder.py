import result
from pathlib import Path

# gather images
imagedirectory = Path('image-loader')
data = []
for images in imagedirectory.iterdir():
    imagename = images.name

    # filter out auxiliary ISIC dataset files
    if imagename.endswith('_superpixels.png'):
        continue
    elif imagename.endswith('_metadata.csv'):
        continue
    else:
        data.append(str(images))

# iterating through all 120 scenarios
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
        for k in range(0,4):
            if k == 0:
                edg = 'Sobel'
            elif k == 1:
                edg = 'Prewitt'
            elif k == 2:
                edg = 'LoG'
            elif k == 3:
                edg = 'Canny'
            code = f'{i}{j}{k}'
            # skip already completed codes
            if code in ['000','010','020','030','040','100','110','120','001','011','002','003','012','013',
                        '021','022','023','031','032','033','041','042','043','101','102','103','111','112',
                        '113','121','122','123','130','131','132','133','140','141','142','143','200','201',
                        '202','203','210','211','212','220','221','222','230','231','232','240','510','511',
                        '512','520','521','522','530','531','532','540','541','542']:
                continue
            if k == 3:
                continue
            print(f'Code: {code}')
            result.calculate(data,pre,deg,edg)