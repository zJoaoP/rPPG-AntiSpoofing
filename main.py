from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose

from utils.argument_parser import MyArgumentParser
from utils.landmarks import LandmarkPredictor
import cv2

if __name__ == "__main__":
	args, wrapper, strategy = MyArgumentParser().parse_args()
	predictor, extractor = LandmarkPredictor(), CheeksAndNose()

	frame_rate = wrapper.get_frame_rate()
	frames_to_skip = 0

	frame_count, frame_limit = 0, args.time_limit * frame_rate
	while frame_count < frame_limit:
		(success, frame) = wrapper.get_frame()
		if not success or cv2.waitKey(1) & 0xFF == ord('q'):
			break
		elif frames_to_skip > 0:
			frames_to_skip -= 1
			continue
		else:
			face_rect = predictor.detect_face(image = frame)
			if face_rect is not None:
				landmarks = predictor.detect_landmarks(image = frame, rect = face_rect)
				frame = extractor.extract_roi(frame, landmarks)

				strategy.process(frame)

				cv2.imshow("rPPG Tracker", frame)
				frame_count += 1
			else:
				frames_to_skip = args.skip_count	

	strategy.show_results()