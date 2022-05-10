from mmkfeatures.video.video_toolkit import VideoEmotionToolkit

video_toolkit = VideoEmotionToolkit()
video_toolkit.empty_result()

path = "deep_face_analysis/video/test.mp4"
video_toolkit.analyze(path, time_thre=5)
result = video_toolkit.get_result()

print()
for item in result:
    print(item[0], item[1])