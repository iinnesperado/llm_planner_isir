import cv2

vidcap = cv2.VideoCapture("result.mp4")
# get total number of frames
totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
randomFrameNumber=0
# set frame position
vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
success, image = vidcap.read()
if success:
    cv2.imwrite("random_frame.jpg", image)
