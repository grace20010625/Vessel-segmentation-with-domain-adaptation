import numpy as np, struct, gzip

def read_nii_gz(path):
    with gzip.open(str(path), 'rb') as f:
        hdr = f.read(352); raw = f.read()
    dim = struct.unpack_from('<8h', hdr, 40)
    shape = tuple(dim[1:dim[0]+1])
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape, order='C').astype(np.float32)

# 路径改成你自己的
img = read_nii_gz('./data/raw/mv02.nii.gz')
cap = float(np.percentile(img, 99.9))
vol_clip = np.clip(img, img.min(), cap)
mu, std = float(vol_clip.mean()), float(vol_clip.std())
img_norm = (vol_clip - mu) / std

npz = np.load('./data/processed/mv02_processed.npz', allow_pickle=True)
images = npz['images']   # (Z, C, H, W)
k = images.shape[1] // 2   # centre channel

print(f"npz shape: {images.shape}")
print(f"npz z=0 mean:  {images[0, k].mean():.4f}")
print(f"norm z=0 mean: {img_norm[:,:,0].mean():.4f}")
print(f"Values match: {np.allclose(images[0, k], img_norm[:,:,0], atol=1e-4)}")