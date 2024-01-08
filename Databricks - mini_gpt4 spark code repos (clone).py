# Databricks notebook source
# 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

# COMMAND ----------

pip install --upgrade pip

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Git LFS installation

# COMMAND ----------

!pip list

# COMMAND ----------

!sudo apt update
!sudo apt install git

# COMMAND ----------

!git --version

# COMMAND ----------

!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

# COMMAND ----------

! apt-get install git-lfs

# COMMAND ----------

!git lfs version

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Load Llama 2 model from Huggingface

# COMMAND ----------

!git lfs install
!git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf /Workspace/Repos/sameer.sharma@meesho.com/MiniGPT-4/Llama-2-7b-chat-hf/

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Vicuna

# COMMAND ----------

!git lfs install
!git clone https://huggingface.co/Vision-CAIR/vicuna-7b

# COMMAND ----------

!git lfs help smudge

# COMMAND ----------

!git clone https://github.com/sameersharma00747/MiniGPT-4.git

# COMMAND ----------

ls Llama-2-7b-chat-hf

# COMMAND ----------

!pip install gradio
!pip install omegaconf
!pip install iopath
!pip install timm
!pip install webdataset
!pip install transformers
!pip install peft
!pip install decord
!pip install SentencePiece
!pip install accelerate
!pip install bitsandbytes

# COMMAND ----------

cd MiniGPT-4

# COMMAND ----------

!pip install numpy==1.23

# COMMAND ----------

!pip install git+https://github.com/huggingface/transformers.git

# COMMAND ----------

!pip install sentencepiece --upgrade

# COMMAND ----------

!python demo.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0

# COMMAND ----------



# COMMAND ----------

# !pip install --upgrade numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

cd MiniGPT-4

# COMMAND ----------

pwd

# COMMAND ----------

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# COMMAND ----------

!cp /dbfs/sameer/catalogv2/final_merged_path_u2net_first_image_1000/277548309.png test_image.png

# COMMAND ----------

ls

# COMMAND ----------

# !python minigpt4_test.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --image-path test_image.png --user-input "Your user input message"

# COMMAND ----------

ls

# COMMAND ----------

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
from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2, CONV_VISION_Vicuna0
from minigpt4.models import *

# Define your variables
cfg_path = "eval_configs/minigpt4_llama2_eval.yaml"
gpu_id = 0
image_path = "path_to_your_image.jpg"
user_input = "Your user input message"
options = None  # Set to None or provide overrides as a list of strings in the format ["option1=value1", "option2=value2"]

# Assign a value to image_path
image_path = "path_to_your_image.jpg"

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description="MiniGPT-4 CLI", add_help=False)
    
    parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=gpu_id, help="specify the GPU to load the model.")
    parser.add_argument("--image-path", default=image_path, help="path to the image you want to use.")
    parser.add_argument("--user-input", default=user_input, help="user input message for chatbot.")
    parser.add_argument("--options", default=options, nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file.")
    
    args, _ = parser.parse_known_args()

    cfg = Config(args)
    
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    
    CONV_VISION = CONV_VISION_LLama2
    
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    
    # Load the image
    image = Image.open(args.image_path)
    
    # Start the conversation
    chat_state = CONV_VISION.copy()
    img_list = []
    
    # Upload the image
    chat.upload_img(image, chat_state, img_list)
    
    # Start the chat with user input
    chat.ask(args.user_input, chat_state)
    
    # Generate a response
    response = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
    
    print("Chatbot Response:")
    print(response)

if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### LLAMA model inference

# COMMAND ----------

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

# Define your variables
cfg_path = "eval_configs/minigpt4_llama2_eval.yaml"
# cfg_path = "eval_configs/minigpt4_eval.yaml"
gpu_id = 0
image_path = "test_image.png"
user_input = "Your user input message"
options = None  # Set to None or provide overrides as a list of strings in the format ["option1=value1", "option2=value2"]

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def load_model(cfg_path, gpu_id, args):
    # Load the model configuration
    cfg = Config(args)
    
    # Configure and initialize the model
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))
    
    # Configure the conversation and processor
    CONV_VISION = CONV_VISION_LLama2
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # Create and return the chat instance
    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
    return chat, CONV_VISION

def perform_inference(chat, CONV_VISION, image_path, user_input):
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
    
    return response

def main():
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description="MiniGPT-4 CLI", add_help=False)
    
    parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=gpu_id, help="specify the GPU to load the model.")
    parser.add_argument("--image-path", default=image_path, help="path to the image you want to use.")
    parser.add_argument("--user-input", default=user_input, help="user input message for chatbot.")
    parser.add_argument("--options", default=options, nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file.")
    
    args, _ = parser.parse_known_args()
    
    # Load the model
    chat, CONV_VISION = load_model(args.cfg_path, args.gpu_id, args)
    
    return chat, CONV_VISION, args

def run_chat(chat, CONV_VISION, args):
    # Perform inference
    response = perform_inference(chat, CONV_VISION, args.image_path, args.user_input)
    
    print("Chatbot Response:")
    print(response)

if __name__ == "__main__":
    chat, CONV_VISION, args = main()
    run_chat(chat, CONV_VISION, args)

# COMMAND ----------

prompt = 'I am a seller on an ecommerce platform. I want to list the saree shown in the image on their website to sell it. Can you generate a product title and a professional product description for me.'

perform_inference(chat, CONV_VISION, args.image_path, prompt)

# COMMAND ----------

from PIL import Image

# COMMAND ----------

Image.open("test_image.png")

# COMMAND ----------

def perform_inference_new(chat, CONV_VISION, image_paths, user_inputs):
    # Load the images
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Start the conversation for each input
    chat_states = [CONV_VISION.copy() for _ in range(len(images))]
    img_lists = [[] for _ in range(len(images))]
    
    # Upload the images for each input
    for i, image in enumerate(images):
        chat.upload_img(image, chat_states[i], img_lists[i])
    
    # Start the chat with user inputs for each input
    for i, user_input in enumerate(user_inputs):
        chat.ask(user_input, chat_states[i])
    
    # Generate responses for each input
    responses = []
    for i in range(len(images)):
        response = chat.answer(conv=chat_states[i], img_list=img_lists[i], num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
        responses.append(response)
    
    return responses

# COMMAND ----------

img_paths = '/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/2_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/1_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/2ug6e_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/7uq7f_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/anarkali_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/auaj0_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/axd3j_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/crape_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/dazzq_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/earring_1.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/fnvoz_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/ihc09_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/italian_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/jcauc_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/odtma_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/pi9a0_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/pillow_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/r46or_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/saree_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/wkgff_512.jpeg,/dbfs/FileStore/shared_uploads/sameer.sharma@meesho.com/minigpt4_test_images/xljeh_512.jpeg'

# COMMAND ----------

img_paths_list = img_paths.split(',')

# COMMAND ----------

len(img_paths_list)

# COMMAND ----------

prompt_list = ['I am a seller on an ecommerce platform. I want to list the product shown in the image on their website to sell it. Can you generate a product title and a professional product description for me.']*21

# COMMAND ----------

prompt_list

# COMMAND ----------

perform_inference_new(chat, CONV_VISION, img_paths_list, prompt_list)

# COMMAND ----------

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COMMAND ----------

import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
else:
    print("CUDA is not available. No GPUs found.")


# COMMAND ----------

10.73/21

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Multi-GPU inferencing

# COMMAND ----------

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

# Define your variables
cfg_path = "eval_configs/minigpt4_llama2_eval.yaml"
gpu_ids = [0, 1, 2, 3]  # List of GPU IDs
image_paths = img_paths_list  # List of image paths
user_inputs = prompt_list  # List of user inputs
options = None  # Set to None or provide overrides as a list of strings in the format ["option1=value1", "option2=value2"]

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def load_model(cfg_path, gpu_ids, args):
    # Load the model configuration
    cfg = Config(args)
    
    # Configure and initialize the model
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_ids[0]  # Use the first GPU for model initialization
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_ids[0]))
    
    # Configure the conversation and processor
    CONV_VISION = CONV_VISION_LLama2
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # Create and return the chat instance
    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_ids[0]))
    
    # Use DataParallel to parallelize the model across GPUs
    if len(gpu_ids) > 1:
        chat = torch.nn.DataParallel(chat, device_ids=gpu_ids)
    
    return chat, CONV_VISION

def perform_inference(chat, CONV_VISION, image_paths, user_inputs):
    # Load the images
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Start the conversation for each input
    chat_states = [CONV_VISION.copy() for _ in range(len(images))]
    img_lists = [[] for _ in range(len(images))]
    
    # Upload the images for each input
    for i, image in enumerate(images):
        chat.module.upload_img(image, chat_states[i], img_lists[i])
    
    # Start the chat with user inputs for each input
    for i, user_input in enumerate(user_inputs):
        chat.module.ask(user_input, chat_states[i])
    
    # Generate responses for each input
    responses = []
    for i in range(len(images)):
        response = chat.module.answer(conv=chat_states[i], img_list=img_lists[i], num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
        responses.append(response)
    
    return responses


# Create an ArgumentParser
parser = argparse.ArgumentParser(description="MiniGPT-4 CLI", add_help=False)

parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
parser.add_argument("--gpu-ids", default=gpu_ids, nargs="+", type=int, help="list of GPU IDs.")
parser.add_argument("--image-paths", default=image_paths, nargs="+", help="list of image paths.")
parser.add_argument("--user-inputs", default=user_inputs, nargs="+", help="list of user inputs.")
parser.add_argument("--options", default=options, nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file.")

args, _ = parser.parse_known_args()

# Load and parallelize the model
chat, CONV_VISION = load_model(args.cfg_path, args.gpu_ids, args)

# Perform inference on a batch of inputs
# responses = perform_inference(chat, CONV_VISION, args.image_paths, args.user_inputs)

# print("Chatbot Responses:")
# for i, response in enumerate(responses):
#     print(f"Response {i+1}:")
#     print(response)

# COMMAND ----------

def perform_inference(chat, CONV_VISION, image_paths, user_inputs):
    # Load the images
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Start the conversation for each input
    chat_states = [CONV_VISION.copy() for _ in range(len(images))]
    img_lists = [[] for _ in range(len(images))]
    
    # Upload the images for each input
    for i, image in enumerate(images):
        chat.module.upload_img(image, chat_states[i], img_lists[i])
    
    # Start the chat with user inputs for each input
    for i, user_input in enumerate(user_inputs):
        chat.module.ask(user_input, chat_states[i])
    
    # Generate responses for each input
    responses = []
    for i in range(len(images)):
        response = chat.module.answer(conv=chat_states[i], img_list=img_lists[i], num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
        responses.append(response)
    
    return responses

# COMMAND ----------

responses = perform_inference(chat, CONV_VISION, img_paths_list_new, args.user_inputs)

# COMMAND ----------

args.image_paths

# COMMAND ----------

type(img_paths)

# COMMAND ----------

img_paths

# COMMAND ----------

import shutil

img_paths_list_new = []

for files in img_paths_list:
    shutil.copy(files, files.split('/')[-1])

    img_paths_list_new.append(files.split('/')[-1])

# COMMAND ----------

img_paths_list_new

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### VICUNA model inference

# COMMAND ----------

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
from minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2, CONV_VISION_Vicuna0
from minigpt4.models import *

# Define your variables
# cfg_path = "eval_configs/minigpt4_llama2_eval.yaml"
cfg_path = "eval_configs/minigpt4_eval.yaml"
gpu_id = 0
image_path = "test_image.png"
user_input = "Your user input message"
options = None  # Set to None or provide overrides as a list of strings in the format ["option1=value1", "option2=value2"]

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def load_model(cfg_path, gpu_id, args):
    # Load the model configuration
    cfg = Config(args)
    
    # Configure and initialize the model
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))
    
    # Configure the conversation and processor
    CONV_VISION = CONV_VISION_Vicuna0
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # Create and return the chat instance
    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
    return chat, CONV_VISION

def perform_inference(chat, CONV_VISION, image_path, user_input):
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
    
    return response

def main():
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description="MiniGPT-4 CLI", add_help=False)
    
    parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=gpu_id, help="specify the GPU to load the model.")
    parser.add_argument("--image-path", default=image_path, help="path to the image you want to use.")
    parser.add_argument("--user-input", default=user_input, help="user input message for chatbot.")
    parser.add_argument("--options", default=options, nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file.")
    
    args, _ = parser.parse_known_args()
    
    # Load the model
    chat, CONV_VISION = load_model(args.cfg_path, args.gpu_id, args)
    
    return chat, CONV_VISION, args

def run_chat(chat, CONV_VISION, args):
    # Perform inference
    response = perform_inference(chat, CONV_VISION, args.image_path, args.user_input)
    
    print("Chatbot Response:")
    print(response)

if __name__ == "__main__":
    chat, CONV_VISION, args = main()
    run_chat(chat, CONV_VISION, args)

# COMMAND ----------

prompt = 'I am a seller on an ecommerce platform. I want to list the saree shown in the image on their website to sell it. Can you generate a product title and a professional product description for me.'

perform_inference(chat, CONV_VISION, args.image_path, prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Spark code

# COMMAND ----------

import sys

import re
import pyspark.sql.types as T
import pyspark.sql.functions as f
import pyspark.sql.functions as func
import pickle
import urllib.request
import datetime, time
import multiprocessing
import os
import io
import boto3
import numpy as np
import pandas as pd
import subprocess
import pytz
import matplotlib.pyplot as plt
import urllib
import cv2
from datetime import datetime, date

from tqdm import tqdm
from itertools import repeat
from PIL import Image
from multiprocessing import Pool

# COMMAND ----------

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

# Define your variables
cfg_path = "eval_configs/minigpt4_llama2_eval.yaml"
# cfg_path = "eval_configs/minigpt4_eval.yaml"
gpu_id = 0
image_path = "test_image.png"
user_input = "Your user input message"
options = None  # Set to None or provide overrides as a list of strings in the format ["option1=value1", "option2=value2"]

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def load_model(cfg_path, gpu_id, args):
    # Load the model configuration
    cfg = Config(args)
    
    # Configure and initialize the model
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))
    
    # Configure the conversation and processor
    CONV_VISION = CONV_VISION_LLama2
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # Create and return the chat instance
    chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
    return chat, CONV_VISION

def main():
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description="MiniGPT-4 CLI", add_help=False)
    
    parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=gpu_id, help="specify the GPU to load the model.")
    parser.add_argument("--image-path", default=image_path, help="path to the image you want to use.")
    parser.add_argument("--user-input", default=user_input, help="user input message for chatbot.")
    parser.add_argument("--options", default=options, nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file.")
    
    args, _ = parser.parse_known_args()
    
    # Load the model
    chat, CONV_VISION = load_model(args.cfg_path, args.gpu_id, args)
    
    return chat, CONV_VISION, args

def run_chat(chat, CONV_VISION, args):
    # Perform inference
    response = perform_inference(chat, CONV_VISION, args.image_path, args.user_input)
    
    print("Chatbot Response:")
    print(response)


chat, CONV_VISION, args = main()
# run_chat(chat, CONV_VISION, args)

# COMMAND ----------

# export

def get_first_valid_img_link(all_img_links_str):
    """
    given a string of comma seperated image links, this func returns the first valid img link
    if no valid link exists, return empty string
    """
    valid_links = re.findall(
        r"(\/images\/products\/[0-9]+\/[\w]+\.jpg)", all_img_links_str
    )
    if len(valid_links):
        return valid_links[0]
    else:
        return ""

def get_metadata_query():

    query1 = """
    select
        a.id as product_id,
        a.images as all_image_links,
        a.image_512 as all_image_links_512,
        d.subsubcategoryid as sscat_id,
        concat_ws(' ', fs.easyocr_text, fs.lavis_text) as ocr,
        fs.ocr_text_0,
        fs.ocr_text_90,
        fs.ocr_text_180,
        fs.ocr_text_270 
    from
        silver.supply__products a
        left join silver.feature_store fs on fs.product_id = a.id
        left join silver.taxonomy__taxonomy_content_sub_sub_category_map d on a.id = d.contentid
        left anti join silver.watermark_ds id on id.product_id = a.id
    where
        a.created > DATEADD(day,-2, GETDATE())
        --and fs.easyocr_text is not null

    limit 100
    """
    
    query2 = """
    select 
            a.name as l4_name
            , b.name as l3_name
            ,c.name as l2_name
            ,d.name as l1_name
            ,a.id as sscat_id
            ,b.id as l3
            , c.id as l2
            ,d.id as l1 
    from silver.taxonomy__taxonomy_sub_sub_category a 
    left join silver.taxonomy__taxonomy_sub_category b on a.sub_category_id = b.id 
    left join silver.taxonomy__taxonomy_category c on b.categoryid=c.id 
    left join silver.taxonomy__taxonomy_super_category d on d.id=c.supercategoryid
    """

    pdf1 = spark.sql(query1)
    pdf2 = spark.sql(query2)
    pdf = pdf1.join(pdf2,'sscat_id','left')
    pdf = pdf.select(func.col("product_id"), \
                     func.col("all_image_links"), \
                     func.col("all_image_links_512"), \
                     func.col("sscat_id"), \
                     func.col("ocr"), \
                     func.col("l1_name"),\
                     func.col("fs.ocr_text_0"), \
                     func.col("fs.ocr_text_90"), \
                     func.col("fs.ocr_text_180"), \
                     func.col("fs.ocr_text_270"))
    
    return pdf

# def processed_pids():
    
#     query = """
#     select
#         product_id
#     from
#         silver.image_deactivations
#     """
    
#     pdf = spark.sql(query)
#     return pdf

# COMMAND ----------

# export

def download_images(row):
    image_link=row['image_link']
    pid=row['product_id']
   
    try:
        urllib.request.urlretrieve(f"https://images.meesho.com{image_link}", f"/tmp/{pid}.jpg")
    except:
        return
    
def fetch_input(image_path, pid):

    image_url = "https://images.meesho.com/"+image_path[1:]
    urllib.request.urlretrieve(image_url, f"/tmp/{pid}_512.jpg")
    
def download_images_512(row):
    image_link=row['image_link_512']
    pid=row['product_id']
    try:
        fetch_input(image_link, pid)
    except:
        print(f'No Image for pid {pid}')

# COMMAND ----------

# export

class ModelWrapperPickable:	
    """	
    This module is required to be able to pass model weights as parameters to the executor code	
    """	
    def __init__(self, model):
        self.model = model
    def __getstate__(self):
        model_str = ''
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tf.keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d
    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            self.model = tf.keras.models.load_model(fd.name)

# COMMAND ----------

# export

def perform_inference(row, chat, CONV_VISION, user_input):

    pid = row['product_id']

    # Load the image
    image = Image.open(f"/tmp/{pid}.jpg")
    
    # Start the conversation
    chat_state = CONV_VISION.copy()
    img_list = []
    
    # Upload the image
    chat.upload_img(image, chat_state, img_list)
    
    # Start the chat with user input
    chat.ask(user_input, chat_state)
    
    # Generate a response
    response = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
    
    return response

# COMMAND ----------

# export

def id_generator(chat, CONV_VISION):
    def id_inference(it):
        
#         import os.path
#         import boto3
#         import cv2
#         import tensorflow as tf
#         import matplotlib.pyplot as plt
#         from skimage.io import imread
        s3_client = boto3.client("s3")
        s3_resource = boto3.resource("s3")


        sys.path.append('/databricks/driver/MiniGPT-4')
        
#         #tqdm.pandas()
        
        ## converts partions into into pandas df
        id_df_part = pd.DataFrame([r.asDict() for r in list(it)])
        print("recieved parts", len(id_df_part))

        ## if input empty, then return empty
        if len(id_df_part) == 0:
            return [{"output": id_df_part}]
        
        id_df_part["image_link"] = id_df_part["all_image_links"].apply(
            get_first_valid_img_link
        )
        id_df_part["image_link_512"] = id_df_part["all_image_links_512"].apply(
            get_first_valid_img_link
        )
        
        id_df_part.apply(download_images, axis=1)
        # id_df_part.apply(download_images_512, axis=1)

        perform_inference(row, chat, CONV_VISION, user_input)

        id_df_part[['description']] = id_df_part.apply(
            perform_inference,
            chat=chat, 
            CONV_VISION=CONV_VISION,
            user_input='I am a seller on an ecommerce platform. I want to list the product shown in the image on their website to sell it. Can you generate a product title and a professional product description for me.',
            axis=1)

        # id_df_part['is_watermark']= 'No'


        return [{"output": id_df_part}]

    return id_inference

# COMMAND ----------

# export

def id_job(chat, CONV_VISION):
    
    ############################################################## 1. Data Retrieval #################################################################
        
    id_pdf = get_metadata_query() \
            .repartition(480) \
            .cache()
    #     processed_pids_pdf = processed_pids()
    #     id_pdf = id_input_pdf.join(processed_pids_pdf,'product_id','left_anti')

    count = id_pdf.count()
    print("Running the job for {} pids".format(count))

    ############################################################## 2. MODEL INFERENCE #################################################################

    # Creating directory to store images if not already created, to be used in profanity_detection_mod()
    # Using exist_ok=True to not throw FileExistsError
    if not os.path.exists('/tmp/inference/img'):
        os.makedirs('/tmp/inference/img', exist_ok=True)
    

    print("Starting inference parallely")
    partition_outputs = id_pdf.rdd.mapPartitions(
        id_generator(chat, CONV_VISION)
    ).collect()

    id_df = pd.concat([i["output"] for i in partition_outputs]).reset_index(
        drop=True
    )
    
    ############################################################## 3. OUTPUT SAVING #################################################################
    
#     id_df.to_csv("id_df.csv", index=False)
    save_time_stamp = (
        datetime.now()
        .astimezone(pytz.timezone("Asia/Kolkata"))
        .replace(microsecond=0)
        .isoformat()
    )

    ## Saves into datalake

    ## exit if empty
    if len(id_df) == 0:
        return id_df

    id_df = id_df.loc[
        :,
        [
            "product_id",
            "image_link",
            "is_watermark",
            "watermark_score",
            'sscat_id',
        ],
    ]
    id_df["created_time"] = pd.to_datetime(save_time_stamp)
    id_df = id_df.astype({'sscat_id':'double', 
                      'watermark_score': 'double'}, errors='ignore')
    
    id_df['watermark_score'] = id_df['watermark_score'].round(2)

    pdf = spark.createDataFrame(id_df) \
            .withColumn('created_time', func.col('created_time').astype(T.TimestampType()))
    
    pdf.dropDuplicates(['product_id']).createOrReplaceTempView("new_tmp_id_table")

    id_update_query = f"""
        MERGE INTO silver.watermark_ds
        USING new_tmp_id_table
        ON silver.watermark_ds.product_id = new_tmp_id_table.product_id
        WHEN MATCHED THEN
        UPDATE SET *
        WHEN NOT MATCHED
        THEN INSERT *
    """

    print("Updating silver.watermark_ds table ")
    # spark.sql(id_update_query)
    
    return id_df

# COMMAND ----------

# export

if __name__ == "__main__":

    s3 = boto3.client("s3")
    id_df = id_job(chat, CONV_VISION)

# COMMAND ----------

ls ../

# COMMAND ----------

pwd

# COMMAND ----------


