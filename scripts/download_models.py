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


def download_with_requests(url, filepath):
    from tqdm import tqdm
    import requests
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

def download(file_url, file_out):
    if 'drive.google.com' in file_url:
        print('downloading with gdown')
        gdown.download(url=file_url, output=file_out)
    else:
        if os.name == 'nt':     # for Windows, use requests to download
            print('downloading with python requests package')
            download_with_requests(file_url, file_out)
        else:                   # simply use wget to download
            print('downloading with wget')
            os.system(f'wget -c {file_url} -O {file_out}')


# download diffusion body models
for js in DIR_JS:
    with open(js, 'r', encoding='utf-8') as f:
        dir_dict = json.load(f)
    for file_url, file_out in dir_dict.items():
        file_dir = os.path.dirname(file_out)
        if not os.access(file_dir, os.F_OK):
            os.makedirs(file_dir)
        download(file_url, file_out)


# download single checkpoints
for js in FILE_JS:
    with open(js, 'r', encoding='utf-8') as f:
        file_dict = json.load(f)
    for file_url, file_out in file_dict.items():
        download(file_url, file_out)
