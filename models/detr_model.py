import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

class ObjectDetector:
    def __init__(self, model_name='facebook/detr-resnet-50', device='gpu'):
        self.model_name = model_name
        self.device = device
        self.processor, self.model = self.initialize_model()

    def initialize_model(self):
        # Check device compatibility
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        
        # Initialize the DETR model and processor
        processor = DetrImageProcessor.from_pretrained(self.model_name)
        model = DetrForObjectDetection.from_pretrained(
            self.model_name, torch_dtype=self.data_type, low_cpu_mem_usage=True
        )
        
        # Move model to the specified device
        model.to(self.device)
        
        # Use half precision if on GPU
        if self.device != 'cpu':
            model.half()
            
        return processor, model

    def detect_objects(self, image):
        # Prepare the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.data_type)
        outputs = self.model(**inputs)
        
        # Process detection outputs
        # Note: Adjust the following code to match the specific output format of your DETR model
        results = [{'scores': output['scores'].detach().cpu().numpy(),
                    'labels': output['labels'].detach().cpu().numpy(),
                    'boxes': output['boxes'].detach().cpu().numpy()} for output in outputs]
        print("DETR RESULTS: ", results)
        
        return results
