import cv2
import os

# Custom import
import videoinput 
import utilscv
import objrecogn as orec

def main():

	cv2.namedWindow('Features')

	str_src = '0:rows=300:cols=400'

	video_input = videoinput.VideoInput(str_src)

	db_dict = orec.loadModelsFromDirectory()

	while True:
		frame = video_input.read()

		detector = cv2.xfeatures2d.SIFT_create(
			nfeatures=250
		)	

		img_in = cv2.cvtColor(
			frame, 
			cv2.COLOR_BGR2GRAY
		)

		img_out = frame.copy()
		kp, desc = detector.detectAndCompute(img_in, None)
		selected_db = db_dict['SIFT']

		if len(selected_db) > 0:
			img_match_mutual = orec.findMatchingMutuosOptimizado(
				selected_db, 
				desc, 
				kp
			)

			min_inliners = int(20)
			projer = float(5)

			best_img, inliners_web_cam, inliners_db = orec.calculateBestImageByNumInliers(
				selected_db, 
				projer, 
				min_inliners
			)

			if not best_img is None:
				orec.calculateAffinityMatrixAndDraw(
					best_img, 
					inliners_db, 
					inliners_web_cam, 
					img_out
				)
 		

		cv2.imshow('Features', img_out)
		ch = cv2.waitKey(5) & 0xFF
		if ch == 27:
			break

	video_input.close()
	cv2.destroyAllWindows()



if __name__ in "__main__":
	main()