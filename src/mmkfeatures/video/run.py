
import os
import multiprocessing
import ffmpeg

if __name__ == '__main__':
    multiprocessing.freeze_support()
    '''
    
    import ffmpeg
    stream = ffmpeg.input(root_path+'/videos/a.mp4')
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, root_path+'/videos/a_out.mp4')
    ffmpeg.run(stream)
    '''

    for line in open("../dataset/input.csv",'r',encoding='utf-8').readlines():
        ls=line.strip().split(",")
        if os.path.exists(ls[1]):
            os.remove(ls[1])

    os.system("python extract.py --csv=../dataset/input.csv --type=2d --batch_size=4 --num_decoding_thread=4")

