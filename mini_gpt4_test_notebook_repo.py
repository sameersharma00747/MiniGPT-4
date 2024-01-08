# Databricks notebook source
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


