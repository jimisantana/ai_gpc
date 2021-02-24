from os import system

model_file_path = 'covid19-2020-gpc__bert_regular.tar.gz'

"""
run this file just once - downloads sample model and un-tars the model
"""

print('downloading model - this will take some time - model is about 400MB')

system(f'wget http://getfiles.adcore.com/{model_file_path}')

system(f'tar -zxvf {model_file_path}')
