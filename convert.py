import glob
import cv2

pngs = glob.glob("eight/*")
print(pngs)

count = 0

def rotate(img, angle):
    rows,cols, _ = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    ninety = rotate(img, 90)
    oneeighty = rotate(img, 180)
    twoseventy = rotate(img, 270)

    augment_photos = [horizontal_img, vertical_img, ninety, oneeighty, twoseventy]
    
    for index in range(len(augment_photos)):
        cv2.imwrite("extra" + str(count) + ".jpg", augment_photos[index])

print("finished.")
