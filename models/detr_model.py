import torch
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection

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
        image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = DetrForObjectDetection.from_pretrained(self.model_name)
        
        # Move model to the specified device
        model.to(self.device)
        
        # Use half precision if on GPU
        if self.device != 'cpu':
            model.half()
            
        return image_processor, model

    def detect_objects(self, image):
        # Prepare the image
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.data_type)
        outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.3:  # adjust threshold as needed
                box = [round(i, 2) for i in box.tolist()]
                detections.append(f"{self.model.config.id2label[label.item()]}")
        
        if detections:
            detection_summary = f"I detected: {', '.join(detections)}"
        else:
            detection_summary = "No significant objects detected."
        
        return detection_summary
