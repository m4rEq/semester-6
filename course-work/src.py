import cv2
import os

class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        #Nombre del fichero
        self.nameFile = nameFile
        #Shape de la imagen
        self.shape = shape
        #Datos binarios de la imagen
        self.imageBinary = imageBinary
        #KeyPoints de la imagen una vez aplicado el algoritmo de detección de features
        self.kp = kp
        #Descriptores de las features detectadas
        self.desc = desc
        #Matchings de la imagen de la base de datos con la imagen de la webcam
        self.matchingWebcam = []
        #Matching de la webcam con la imagen actual de la base de datos.
        self.matchingDatabase = []
    #Permite vaciar los matching calculados con anterioridad, para una nueva imagen
    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

def main():

	sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)

	db = dict([
		('SIFT', [])
	])

	dir_name = 'template'

	for img in os.listdir(dir_name):
		color_img = cv2.imread(dir_name + '/' + str(img))
		curr_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
		kp, desc = sift.detectAndCompute(curr_img, None)
		db['SIFT'].append(ImageFeature(
			img,
			curr_img,
			color_img,
			kp,
			desc	
		))

	
	while True:
		# detector = cv2.xfeature2d.SIFT_create(nfeatures=250)

		img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img_out = frame.copy()



if __name__ in "__main__":
	main()