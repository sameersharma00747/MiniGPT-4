# Databricks notebook source
!pip install palme

# COMMAND ----------

!cp /dbfs/sameer/catalogv2/final_merged_path_u2net_first_image_1000/277548309.png test_image.png

# COMMAND ----------

!pip install torch --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import torch
from palme.model import PalmE
from PIL import Image

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

# img = Image.open('test_image.png')
# text = 'a'

model = PalmE()
output = model(text, img)

# COMMAND ----------


