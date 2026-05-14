import os
import sys
import json
import re
import decord

video_folder = "/home/cse_h2/dzj0055/HYWUVLM-R1/xd_violence/videos/"
event_filepath = "/home/cse_h2/dzj0055/HYWUVLM-R1/xd_violence/xd_test_800.json"

segment_length = sys.maxsize # in seconds basically infinity
dataset = []
fps = 30.
#breakpoint()

"""
    video_path = os.path.join(visual_base, info['video'])
    example = {
        "problem": {"question": ""}, # Depends on how many frames are provided
        "video_path": video_path,
        "data_type": task_type,
        "durations": info['duration'],
        "timestamps": info['timestamps'] # List[ Tuple(float, float) ]
        "time_reference": info['time_reference']
    }
"""

test_filenames = []
with open(event_filepath, 'r') as infile:
    data = json.load(infile)
    for d in data:
        video = d['video']
        test_filenames.append(video)
#breakpoint()

for filename in sorted( os.listdir(video_folder) ):
    video_id = filename.strip()
    #breakpoint()
    video_id, extension = os.path.splitext( video_id )
    #breakpoint()
    if video_id in test_filenames:
        continue
    video_type =  "Normal" if '_label_A' in video_id else "Abnormal"
    timestamps = []
    dataset.append({
        "video": filename.strip(),
        "video_type": video_type,
        "duration": None,
        "timestamps": None,
        "time_reference": None
    })
with open('./train_xd.json', 'w') as outfile:
    json.dump( dataset, outfile, indent=4 )