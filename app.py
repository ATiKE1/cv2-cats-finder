import cv2 as cv

cat_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalcatface.xml') # or haarcascade_frontalcatface_extended.xml

img = cv.imread('image.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

if len(cats) > 0:
    print(f"On image {len(cats)} cats!")

    for (x, y, w, h) in cats:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('cv2 Cats', img)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Cats do not finded.")