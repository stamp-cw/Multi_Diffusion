import hashlib

def calculate_md5(file_path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

# 替换为你的 celeba.zip 路径
file_path = rf"D:\Project\Multi_Diffusion\data\celeba\img_align_celeba.zip"
md5_hash = calculate_md5(file_path)
print(f"MD5 Hash: {md5_hash}")