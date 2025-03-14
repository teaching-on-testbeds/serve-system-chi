import os
import io
import base64
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort  # Use ONNX Runtime instead of PyTorch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "food11.onnx")

        # Determine execution provider based on args to Triton
        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        else:
            providers = ["CPUExecutionProvider"]
        
        self.ort_session = ort.InferenceSession(model_path, providers=providers)

    def preprocess(self, image_data):
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)

        if isinstance(image_data, bytes):
            image_data = image_data.decode("utf-8")
            image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(image).unsqueeze(0).numpy()
        return img_tensor

    def execute(self, requests):

        batched_inputs = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            input_data_array = in_tensor.as_numpy()
            # Preprocess each input (resulting in a tensor of shape [1, C, H, W])
            single_input = self.preprocess(input_data_array[0, 0])
            # append it to batch
            batched_inputs.append(single_input)

        # Combine inputs along the batch dimension
        batched_tensor = np.concatenate(batched_inputs, axis=0)

        # Run inference on the batch
        ort_inputs = {self.ort_session.get_inputs()[0].name: batched_tensor}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        output_probs = ort_outputs[0]  # Extract softmax probabilities

        # Process the outputs and split them for each request
        classes = np.array([
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"
        ])
        responses = []
        for i, request in enumerate(requests):
            output = output_probs[i:i+1]  
            predicted_class_idx = np.argmax(output)
            predicted_label = classes[predicted_class_idx]
            probability = float(output[0][predicted_class_idx])  

            out_label_np = np.array([[predicted_label]], dtype=object)
            out_prob_np = np.array([[probability]], dtype=np.float32)

            out_tensor_label = pb_utils.Tensor("FOOD_LABEL", out_label_np)
            out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_label, out_tensor_prob]
            )
            responses.append(inference_response)

        return responses

