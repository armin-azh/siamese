import numpa as np


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


for f_path in images_path:
    image = Image.open(f_path)
    image = image.convert('RGB')
    im_pixels = np.asarray(image)
    results = im_darken(im_pixels)
            # tm_im = self.extract_face(tm_im, ent['box'])
    tm_im.save(os.path.join(save_path, name_fmt(str(f_cnt))))
    if cnt % step == 0 and cnt > 0:
        print(f'$ {str(cnt)} faces has saved.')
    cnt += 1
    f_cnt += 1
print(f'$ totally {cnt} faces had been extracted.')