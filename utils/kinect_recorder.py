from wrappers import Kinect_Wrapper

import cv2

if __name__ == "__main__":
	stream = Kinect_Wrapper(width = 1920, height = 1080)
	frame_count, frame_rate = 0, stream.get_frame_rate()
	record_time = 30

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	output = cv2.VideoWriter('output.avi',fourcc, 30.0, (1920, 1080))

	while frame_count < frame_rate * record_time:
		(success, frame) = stream.get_frame()

		if not success or cv2.waitKey(1) & 0xFF == ord('q'):
			break

		cv2.imshow("Kinect Recorder", frame[:, :, :3])
		output.write(frame)
		print(frame.shape)

		frame_count += 1

	output.release()
	cv2.destroyAllWindows()