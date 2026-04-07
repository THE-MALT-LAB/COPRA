import sys
import json
duration_filepath = "/home/cse_h2/dzj0055/HYWUVLM-R1/ucf_crime/annotations/UCF_eval_vera.json"
event_filepath = "/home/cse_h2/dzj0055/HYWUVLM-R1/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
duration_dict = {}
dataset = []
segment_length = sys.maxsize # in seconds essentially infinity
fps = 30.
with open(duration_filepath, 'r') as infile:
    data = json.load(infile)
    for d in data:
        video_id = d['video'].split('/')[-1].replace('.mp4', '')
        video_num_frames = d['length']
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
with open(event_filepath, 'r') as infile:
    for line in infile.readlines():
        d = line.strip().split()
        video = d[0].strip()
        video_id = video.split(".")[0] # Before file extension
        video_type = d[1].strip()
        start0_f, end0_f = int(d[2]), int(d[3])
        start0_s, end0_s = start0_f / fps, end0_f / fps
        start1_f, end1_f = int(d[4]), int(d[5])
        start1_s, end1_s = start1_f / fps, end1_f / fps
        print(video, "processed. Video type", video_type, "events at (", start0_f, ",", end0_f, ") and (", start1_f,",",end1_f,")")
        duration = duration_dict[video_id]
        for clip_start in range(0, int(duration), segment_length): # int takes floor
            clip_end = min( duration, clip_start + segment_length )
            time_reference = f"{clip_start}-->{clip_end}"
            timestamps = []
            if ( start0_s >= 0 ) and ( start0_s < clip_end ) and (end0_s > clip_start):
                timestamps.append( [ 
                    max( clip_start, start0_s ) - clip_start, 
                    min( clip_end, end0_s ) - clip_start
                ]  )
            if ( start1_s >= 0 ) and ( start1_s < clip_end ) and (end1_s > clip_start):
                timestamps.append( [ 
                    max( clip_start, start1_s ) - clip_start, 
                    min( clip_end, end1_s ) - clip_start
                ]  )
            dataset.append({
                "video": video,
                "video_type": video_type,
                "duration": duration,
                "timestamps": timestamps,
                "time_reference": time_reference
            })
with open('./test.json', 'w') as outfile:
    json.dump( dataset, outfile, indent=4 )