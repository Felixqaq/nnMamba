"""Crop the header + spirometry block out of each PFT page for reading.

One crop carries everything needed: the patient block (sex, age, height, weight)
and the Spirometry table with Ref / Pre / Post columns. Coordinates are fractions
of the page so the two page sizes in this archive (3308x4678 and 2481x3508) crop
to the same content.
"""
import os, sys, glob
from PIL import Image

SRC = "/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG"
OUT = "/home/felix/pft_crops"
BATCHES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["20260702", "20260709", "20260716"]
TOP, BOT = 0.055, 0.445
WIDTH = 1500

os.makedirs(OUT, exist_ok=True)
n = 0
for b in BATCHES:
    for p in sorted(glob.glob(f"{SRC}/{b}/*.jpg")):
        pid = os.path.basename(p)[:-4]
        im = Image.open(p)
        W, H = im.size
        c = im.crop((0, int(TOP * H), W, int(BOT * H))).convert("L")
        c = c.resize((WIDTH, int(WIDTH * c.height / c.width)), Image.LANCZOS)
        c.save(f"{OUT}/{b}_{pid}.png")
        n += 1
print(f"{n} crops -> {OUT}")
