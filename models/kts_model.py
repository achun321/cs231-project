import torch
import numpy as np
from models.kts_src.kts_utils import cpd_auto, l2_normalize_np_array

class VideoSegmentor:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, video_features, video_length):
        frame_count = len(video_features)
        seg_windows = [[i, i + 1] for i in range(frame_count - 1)]
        return seg_windows
