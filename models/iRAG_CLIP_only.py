import os
import cv2
import pdb
import sys
import time
import torch
import numpy as np
from transformers import logging
logging.set_verbosity_error()

from models.kts_model import VideoSegmentor
from models.irag_clip_model import FeatureExtractor
from models.blip2_model import ImageCaptioner
from models.detr_model import ObjectDetector
from models.grit_model import DenseCaptioner
from models.whisper_model import AudioTranslator
from models.gpt_model_clip_only import LlmReasoner
from utils.utils import logger_creator, format_time


class iRAGCLIP:
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.data_dir = args.data_dir
        self.tmp_dir = args.tmp_dir
        self.models_flag = False
        self.init_llm()
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        
    def init_models(self): 
        print('\033[1;34m' + "Welcome to the our Vlog toolbox...".center(50, '-') + '\033[0m')
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        print('\033[1;31m' + "This may time-consuming, please wait...".center(50, '-') + '\033[0m')
        # At the start of the feature extraction
        torch.cuda.reset_peak_memory_stats(device='cuda')

        print("CLIP ONLY")
        self.feature_extractor = FeatureExtractor(self.args)
        self.video_segmenter = VideoSegmentor(alpha=self.alpha, beta=self.beta)
        # self.object_detector = ObjectDetector(model_name=self.args.object_detector, device=self.args.object_detector_device)
        #elf.image_captioner = ImageCaptioner(model_name=self.args.captioner_base_model, device=self.args.image_captioner_device)
        #self.dense_captioner = DenseCaptioner(device=self.args.dense_captioner_device)
        # self.audio_translator = AudioTranslator(model=self.args.audio_translator, device=self.args.audio_translator_device)
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')
    
    def init_llm(self):
        print('\033[1;33m' + "Initializing LLM Reasoner...".center(50, '-') + '\033[0m')
        os.environ["OPENAI_API_KEY"] = self.args.openai_api_key
        self.llm_reasoner = LlmReasoner(self.args)
        print('\033[1;32m' + "LLM initialization finished!".center(50, '-') + '\033[0m')

    def video2log(self, video_path): 
        video_path = video_path
        video_id = os.path.basename(video_path).split('.')[0]
        if self.llm_reasoner.exist_vectorstore(video_id):
            return 
        try:
            self.llm_reasoner.create_vectorstore(video_id)
        except:
            pass        

        if not self.models_flag:
            self.init_models()
            self.models_flag = True

        start_time = time.time()
        clip_features, video_length = self.feature_extractor(video_path, video_id)

        self.llm_reasoner.create_vectorstore(video_id)
        end_time = time.time()
        print(f"Time to create vector store: {end_time - start_time:.2f} seconds")
        # After the feature extraction
        print(f"Max memory allocated on GPU: {torch.cuda.max_memory_allocated(device='cuda') / 1024**2:.2f} MB")
    
    def chat2video(self, user_input):
        response = self.llm_reasoner(user_input)
        return response

    def clean_history(self):
        self.llm_reasoner.clean_history()
        return
