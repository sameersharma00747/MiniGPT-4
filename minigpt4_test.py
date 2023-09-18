import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2
from minigpt4.models import *

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 CLI")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the GPU to load the model.")
    parser.add_argument("--image-path", required=True, help="path to the image you want to use.")
    parser.add_argument("--user-input", type=str, required=True, help="user input message for chatbot.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into the config file.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    args = parse_args()
    cfg = Config(args)
    
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    
    CONV_VISION = CONV_VISION_LLama2
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    
    image_path = args.image_path
    user_input = args.user_input
    
    # Load the image
    image = Image.open(image_path)
    
    # Start the conversation
    chat_state = CONV_VISION.copy()
    img_list = []
    
    # Upload the image
    chat.upload_img(image, chat_state, img_list)
    
    # Start the chat with user input
    chat.ask(user_input, chat_state)
    
    # Generate a response
    response = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
    
    print("Chatbot Response:")
    print(response)

if __name__ == "__main__":
    main()
