from scenedetect import detect, ContentDetector


scene_list = detect('/home/adamchun/cs231n-project/data/VQAv2/imbalanced_output_video.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].get_frames(),
        scene[1].get_timecode(), scene[1].get_frames(),))