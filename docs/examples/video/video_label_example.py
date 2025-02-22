#!/bin/env python

import os
import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

try:
    from matplotlib import pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

try:
    from PIL import Image
    has_PIL = True
except ImportError:
    has_PIL = False

BATCH_SIZE=4
COUNT=5

def YUV2RGB(yuv):
    yuv = np.multiply(yuv, 255)
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    return rgb

VIDEO_FILE_ROOT = "labelled_videos"

ITER=100

class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.VideoReader(device="gpu", file_root = data, sequence_length=COUNT,
        # instead of file_root, path to text file with pairs video_filepath label_value can be provided
        # self.input = ops.VideoReader(device="gpu", file_list = "file_list.txt", sequence_length=COUNT,
                                     shard_id=0, num_shards=1, random_shuffle=False,
                                     normalized=True, image_type=types.YCbCr, dtype=types.FLOAT)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

if __name__ == "__main__":
    pipe = VideoPipe(batch_size=BATCH_SIZE, num_threads=2, device_id=0, data=VIDEO_FILE_ROOT)
    pipe.build()
    for i in range(ITER):
        print("Iteration " + str(i))
        sequences_out, label = pipe.run()
        sequences_out = sequences_out.as_cpu().as_array()
        label  = label.as_cpu().as_array()
        print("sequences shape: ", sequences_out.shape)
        print("labels shape: ", label.shape)
        print("Got sequence " + str(i*COUNT) + " " + str((i + 1)*COUNT - 1))

        for b in range(BATCH_SIZE):
            batch_sequences = sequences_out[b]
            print(label[b])
            print(batch_sequences.shape)
            save_dir = 'extracted_frames/' + str(label[b][0]) + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for c in range(COUNT):
                sample_frame = batch_sequences[c]
                if has_PIL:
                    im = Image.fromarray(YUV2RGB(sample_frame).astype('uint8'))
                    im.save(save_dir + str(i * BATCH_SIZE * COUNT + b * COUNT + c) + '.png')

    frame_to_show = sequences_out[0][0]
    frame_to_show = YUV2RGB(frame_to_show)

    if has_matplotlib:
        plt.imshow(frame_to_show.astype('uint8'), interpolation='bicubic')
        plt.show()
        plt.savefig('saved_frame.png')

