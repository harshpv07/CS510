import torch
from PIL import Image
import os
import traceback
from vlm import load_image, split_model
from transformers import AutoModel, AutoTokenizer
import datetime



class generateReponse:
    def __init__(self):

        # Initialize model and tokenizer
        self.path = 'OpenGVLab/InternVL2_5-1B'
        self.device_map = split_model('InternVL2_5-1B')
        self.model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=False)

    def get_response(self, image_path,  base_prompt):
        
        """
        Process an image file and a prompt to get a response from the VLM model.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): User prompt text
            
        Returns:
            dict: A dictionary containing status and response or error information
        """
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                return {
                    'status': 'error',
                    'error': f'Image file not found: {image_path}'
                }
                
            # Process image using VLM model
            try:
                # Convert image to tensor
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=1024, do_sample=True)
                
                # Add image token to prompt
               
                full_prompt = f'{base_prompt}\n<image>\n'
                
                # Get response from model
                response = self.model.chat(self.tokenizer, pixel_values, full_prompt, generation_config)
                
                return {
                    'status': 'success',
                    'response': response,
                    'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                    
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"Error processing image: {str(e)}\n{error_traceback}")
                return {
                    'status': 'error',
                    'error': f'Error processing image: {str(e)}',
                    'traceback': error_traceback
                }
                    
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Server error: {str(e)}\n{error_traceback}")
            return {
                'status': 'error',
                'error': f'Server error: {str(e)}',
                'traceback': error_traceback
            }


    def describe_image(self):
        prompt = """

            Describe the emotions on the face of this person is expressing in the provided image in detail. The description should only contain the emotions on the face of the person and nothing else. When answering this question, always return the response in the following format and do not ask any further questions.  
            {
                Base Emotion : <The type of emotion expressed. It could be Happy or Sad or Anxious or anything general> 
                Description : <Detailed Description of the emotion> 
            }

        """
        output = self.get_response("examples/image1.jpg" , prompt) 
        print(output) 

        
gr = generateReponse()
gr.describe_image()