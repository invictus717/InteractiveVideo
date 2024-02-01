import os, json, gdown


def custom_makedirs(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


custom_makedirs('checkpoints/drag')
custom_makedirs('checkpoints/diffusion_body')
custom_makedirs('checkpoints/i2i/lora')
custom_makedirs('checkpoints/i2v/unet')
custom_makedirs('checkpoints/i2v/dreambooth')


# only a single checkpoint is required
FILE_JS = [
    'scripts/i2i_lora.json',
    'scripts/i2v_dreambooth.json',
    'scripts/i2v_unet.json',
    'scripts/drag.json'
]
# complex model directory (diffusion body models)
DIR_JS = [
    'scripts/kohaku-v2.1.json',
    'scripts/stable-diffusion-v1-5.json',
    'scripts/sd-turbo.json'
]


# download diffusion body models
for js in DIR_JS:
    with open(js, 'r', encoding='utf-8') as f:
        dir_dict = json.load(f)
    for file_url, file_out in dir_dict.items():
        file_dir = os.path.dirname(file_out)
        if not os.access(file_dir, os.F_OK):
            os.makedirs(file_dir)
        if 'drive.google.com' in file_url:
            gdown.download(url=file_url, output=file_out)
        else:
            os.system(f'wget -c {file_url} -O {file_out}')


# download single checkpoints
for js in FILE_JS:
    with open(js, 'r', encoding='utf-8') as f:
        file_dict = json.load(f)
    for file_url, file_out in file_dict.items():
        if 'drive.google.com' in file_url:
            gdown.download(url=file_url, output=file_out)
        else:
            os.system(f'wget -c {file_url} -O {file_out}')
