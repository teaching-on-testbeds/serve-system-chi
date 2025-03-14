import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "food11.pth")
        
        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        self.classes = np.array([
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"
        ])



    def preprocess(self, image_data):
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)

        if isinstance(image_data, bytes):
            image_data = image_data.decode("utf-8")
            image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_data)).convert('RGB')


        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor

    def execute(self, requests):
        # Gather inputs from all requests
        batched_inputs = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            input_data_array = in_tensor.as_numpy()  # each assumed to be shape [1]
            # Preprocess each input (resulting in a tensor of shape [1, C, H, W])
            batched_inputs.append(self.preprocess(input_data_array[0, 0]))
        
        # Combine inputs along the batch dimension
        batched_tensor = torch.cat(batched_inputs, dim=0).to(self.device)
        print("BatchSize: ", len(batched_inputs))
        # Run inference once on the full batch
        with torch.no_grad():
            outputs = self.model(batched_tensor)
        
        # Process the outputs and split them for each request
        responses = []
        for i, request in enumerate(requests):
            output = outputs[i:i+1]  # select the i-th output
            prob, predicted_class = torch.max(output, 1)
            predicted_label = self.classes[predicted_class.item()]
            probability = torch.sigmoid(prob).item()
            
            # Create numpy arrays with shape [1, 1] for consistency.
            out_label_np = np.array([[predicted_label]], dtype=object)
            out_prob_np = np.array([[probability]], dtype=np.float32)
            
            out_tensor_label = pb_utils.Tensor("FOOD_LABEL", out_label_np)
            out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_label, out_tensor_prob])
            responses.append(inference_response)
        
        return responses


