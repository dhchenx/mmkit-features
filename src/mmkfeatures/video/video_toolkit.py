import pandas as pd
from deepface import DeepFace

class VideoEmotionToolkit:

    def __init__(self):
        self.elist=[]
        self.count=0
        pass

    def empty_result(self):
        self.elist=[]
        self.count=0

    def call_back(self,emotions,age,gender,timepoint):
        dict_model={}
        for index, row in emotions.iterrows():
            dict_model[str(row["emotion"]).lower()]=row["score"]
        interval=(self.count*self.time_threshold,(self.count+1)*self.time_threshold)
        self.elist.append([timepoint,dict_model])
        self.count+=1

    def get_result(self):
        return self.elist

    def analyze(self,source,time_thre=5):
        self.time_threshold=time_thre
        DeepFace.stream(
            # db_path="../doctor_images",
            # source=0,
            source=source,
            enable_face_recognition=False, call_back_func=self.call_back, time_threshold=time_thre)


if __name__=="__main__":
    video_toolkit = VideoEmotionToolkit()
    video_toolkit.empty_result()

    path = "deep_face_analysis/video/test.mp4"
    video_toolkit.analyze(path, time_thre=5)
    result = video_toolkit.get_result()

    print()
    for item in result:
        print(item[0], item[1])



