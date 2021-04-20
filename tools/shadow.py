import numpy as np
import pathlib
from PIL import Image
import os
from tqdm import tqdm
from recognition.utils import Timer


def im_darken(im):
    dark_im = im.copy()
    h, w, d = im.shape
    alpha = np.random.uniform(0.0, 0.4)
    px = np.random.uniform(0.4, 1.0)
    py = np.random.uniform(0.4, 1.0)
    for y in range(int(h * px)):
        for x in range(int(w * py)):
            for c in range(d):
                dark_im[y, x, c] = np.clip(alpha * im[y, x, c], 0, 255)
    return dark_im


def im_lighten(im):
    dark_im = im.copy()
    h, w, d = im.shape
    alpha = np.random.uniform(2, 10)
    px = np.random.uniform(0.4, 1.0)
    py = np.random.uniform(0.4, 1.0)
    for y in range(int(h * px)):
        for x in range(int(w * py)):
            for c in range(d):
                dark_im[y, x, c] = np.clip(alpha * im[y, x, c], 0, 255)
    return dark_im


def add_shadow(args):
    timer = Timer()
    print("$ start manipulate")
    base = pathlib.Path(args.test_dir)
    save_path = pathlib.Path(args.im_save)
    if not save_path.exists():
        save_path.mkdir()
    timer.start()
    for ch in tqdm(base.glob("**/*.jpg")):
        im = Image.open(ch)
        im_b_s_p = os.path.join(save_path, ch.parent.name)
        if args.im_darker:
            im_b_s_p_d = pathlib.Path(os.path.join(im_b_s_p, ch.stem + '_darker' + ch.suffix))
            if not im_b_s_p_d.parent.exists():
                im_b_s_p_d.parent.mkdir()
            im_d = im_darken(np.asarray(im))
            im_d = Image.fromarray(im_d)
            im_d.save(im_b_s_p_d)
        if args.im_brighter:
            im_b_s_p_b = pathlib.Path(os.path.join(im_b_s_p, ch.stem + '_brighter' + ch.suffix))
            if not im_b_s_p_b.parent.exists():
                im_b_s_p_b.parent.mkdir()
            im_d = im_lighten(np.asarray(im))
            im_d = Image.fromarray(im_d)
            im_d.save(im_b_s_p_b)

    print(f"$ manipulation has been done in {timer.end()}")