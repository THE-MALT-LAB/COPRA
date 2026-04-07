import os
import sys
import json
import re

video_folder = "/home/cse_h2/dzj0055/HYWUVLM-R1/ucf_crime/videos/"
duration_filepath = "/home/cse_h2/dzj0055/HYWUVLM-R1/ucf_crime/annotations/my_all.txt"
event_filepath = "/home/cse_h2/dzj0055/HYWUVLM-R1/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

segment_length = sys.maxsize # in seconds basically infinity
duration_dict = {}
dataset = []
fps = 30.
with open(duration_filepath, 'r') as infile:
    for line in infile.readlines():
        d = line.strip().split()
        video_id = d[0]
        video_start, video_end = int( d[1] ), int( d[2] )
        video_num_frames = video_end - video_start
        duration = video_num_frames / fps
        duration_dict[ video_id ] = duration
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
    for line in infile.readlines():
        d = line.strip().split()
        video = d[0].strip()
        test_filenames.append(video)

for filename in sorted( os.listdir(video_folder) ):
    if filename in test_filenames:
        continue
    video = filename.strip()
    video_id = video.split(".")[0] # Before file extension
    video_type = re.sub( "[0-9]+", "", video.split("_")[0] )
    duration = duration_dict[video_id]
    for clip_start in range(0, int(duration), segment_length): # int takes floor
        clip_end = min( duration, clip_start + segment_length )
        time_reference = f"{clip_start}-->{clip_end}"
        timestamps = []
        dataset.append({
            "video": video,
            "video_type": video_type,
            "duration": duration,
            "timestamps": timestamps,
            "time_reference": time_reference
        })
with open('./train.json', 'w') as outfile:
    json.dump( dataset, outfile, indent=4 )