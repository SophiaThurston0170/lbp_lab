import numpy as np
from scipy import ndimage
from lbpface.datasets import load_orl
from lbpface.features import LBPDescriptor
from lbpface.distances import pairwise_distances
from lbpface.classify import recognition_rate

fs = load_orl("data/ORL")
desc = LBPDescriptor.orl()                          # LBP(16,2), 30x37 windows
is_gallery = (np.arange(len(fs.images)) % 10) < 5   # images 1-5 of each subject
gal, prb = fs.images[is_gallery], fs.images[~is_gallery]
gl, pl = fs.subjects[is_gallery], fs.subjects[~is_gallery]

def normalise(img, sigma=8):     # cancel smooth lighting: divide by a blurred copy
    img = np.asarray(img, dtype=float)
    return img / (ndimage.gaussian_filter(img, sigma) + 1.0)

def rotate(img, deg):
    return ndimage.rotate(img.astype(float), deg, reshape=False, order=1, mode="nearest")

# store every gallery face at several tilts, so a tilted probe finds a tilted match
ANGLES = [-20, -10, 0, 10, 20]
G = np.concatenate([desc.transform([normalise(i if a == 0 else rotate(i, a)) for i in gal]) for a in ANGLES])
GL = np.tile(gl, len(ANGLES))                       # subject label of every stored copy

def score(probe_images):
    P = desc.transform([normalise(i) for i in probe_images])
    return recognition_rate(pairwise_distances(P, G, "chi2"), GL, pl)

def illum_ramp(img, s):          # darker on the left, brighter on the right
    return np.clip(img * (1 + s * np.linspace(-1, 1, img.shape[1])), 0, 255)

print("baseline           ", score(prb))
print("uniform brightness ", score([0.5 * i + 20 for i in prb]))
for s in [0.2, 0.4, 0.6, 0.8]: print("lighting ramp", s, score([illum_ramp(i, s) for i in prb]))
for d in [5, 10, 20]:          print("rotate", d, "deg", score([rotate(i, d) for i in prb]))