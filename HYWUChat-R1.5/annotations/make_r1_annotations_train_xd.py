import os
import sys
import json
import re
import decord

video_folder = "/home/cse_h2/dzj0055/HYWUChat-R1.5/xd_violence/videos/"
duration_filepath = "/home/cse_h2/dzj0055/HYWUChat-R1.5/xd_violence/xd_train.json"
event_filepath = "/home/cse_h2/dzj0055/HYWUChat-R1.5/xd_violence/xd_test_800.json"

segment_length = sys.maxsize # in seconds basically infinity
duration_dict = {}
dataset = []
fps = 24.
#breakpoint()

with open(duration_filepath, 'r') as duration_file:
    data = json.load(duration_file)
    for d in data:
        duration_dict[ d['video_name'] ] = d['length'] / fps

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
    
    if video_id in duration_dict:
        duration = duration_dict[video_id]
        #breakpoint()
        #print(f"Got {duration} for {video_id}")
    else:
        try:
            vr = decord.VideoReader( os.path.join(video_folder, filename), ctx=decord.cpu(0) )
        except:
            print(f"Could not open {video_id}!")
            continue
        duration = len(vr) / fps
        #breakpoint()
        print(f"Got {duration} for {video_id} from decord")


    video_type =  "Normal" if '_label_A' in video_id else "Abnormal"
    timestamps = []
    dataset.append({
        "video": filename.strip(),
        "video_type": video_type,
        "duration": duration,
        "timestamps": None,
        "time_reference": None
    })
with open('./train_xd.json', 'w') as outfile:
    json.dump( dataset, outfile, indent=4 )