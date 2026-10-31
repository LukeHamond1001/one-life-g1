"""REAL PHOTOS FOR THE LANGUAGE WORKSHOPS (docs/LEVELS.md, the language workshops L1-L10; the owner 10-30 evening: "we should use irl
photos and vids mixed with sim stuff to make sure everything being generalized and absorbed"). COCO 2017 (cocodataset.org: every
object outlined and named, five sentences a photo; the annotations CC BY 4.0, each photo under its own Flickr licence, kept on this
machine only, in data/): the annotations fetched once, then for each word the lessons teach, its photos fetched one by one (COCO serves
each at its own address, no archive) and each outlined object cut out square with a margin, at --px a side, into
data/photos/coco/crops/<word>/<annotation id>.jpg, with an index (data/photos/coco/crops/index.json: word, the photo, its licence,
the box, the split: the val photos are never shown in a lesson, they are the tests). Downloaded files are data: read as images and JSON,
nothing run from them.

    python3 -I tools/levels/photos_fetch.py [--per 150] [--test 40] [--px 128] [--min-side 64] [--threads 12]
"""
import argparse
import io
import json
import os
import re
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_FILES = ("annotations/instances_train2017.json", "annotations/instances_val2017.json", "annotations/captions_train2017.json",
             "annotations/captions_val2017.json")
UA = {"User-Agent": "one-life-g1-sim/1.0 (a simulated child robot's picture lessons)"}
WORDS = {   # COCO's name -> the word she says (a home's things, what a toddler is told first; ours)
    "cup": "cup", "bowl": "bowl", "bottle": "bottle", "spoon": "spoon", "fork": "fork", "knife": "knife", "wine glass": "glass",
    "book": "book", "chair": "chair", "couch": "couch", "bed": "bed", "dining table": "table", "tv": "tv", "laptop": "laptop",
    "cell phone": "phone", "remote": "remote", "keyboard": "keyboard", "mouse": "mouse", "clock": "clock", "vase": "vase",
    "potted plant": "plant", "teddy bear": "bear", "sports ball": "ball", "banana": "banana", "apple": "apple", "orange": "orange",
    "cake": "cake", "donut": "donut", "pizza": "pizza", "broccoli": "broccoli", "carrot": "carrot", "sandwich": "sandwich",
    "dog": "dog", "cat": "cat", "bird": "bird", "horse": "horse", "sheep": "sheep", "cow": "cow", "elephant": "elephant",
    "car": "car", "truck": "truck", "bus": "bus", "bicycle": "bike", "boat": "boat", "train": "train", "airplane": "plane",
    "umbrella": "umbrella", "backpack": "backpack", "handbag": "bag", "scissors": "scissors", "toothbrush": "toothbrush",
    "sink": "sink", "toilet": "toilet", "refrigerator": "fridge", "microwave": "microwave", "oven": "oven", "person": "person",
}
_lock = threading.Lock()


def _get(url, tries=3, timeout=60):
    for k_ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r_:
                return r_.read()
        except Exception:
            if k_ == tries - 1:
                raise
            time.sleep(1.5 * (k_ + 1))


def fetch_annotations(root):
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    if all(os.path.exists(os.path.join(root, f_)) for f_ in ANN_FILES):
        return
    zp_ = os.path.join(root, "annotations_trainval2017.zip")
    if not os.path.exists(zp_):
        print(f"fetching {ANN_URL}", flush=True)
        with urllib.request.urlopen(urllib.request.Request(ANN_URL, headers=UA), timeout=120) as r_, open(zp_ + ".part", "wb") as f_:
            while True:
                b_ = r_.read(1 << 20)
                if not b_:
                    break
                f_.write(b_)
        os.replace(zp_ + ".part", zp_)
    with zipfile.ZipFile(zp_) as z_:
        for nm_ in ANN_FILES:                                          # (only the four files named here: nothing else unpacked)
            with z_.open(nm_) as src_, open(os.path.join(root, nm_), "wb") as dst_:
                while True:
                    b_ = src_.read(1 << 20)
                    if not b_:
                        break
                    dst_.write(b_)


def plan(root, per, test, min_side, seed=0):
    """-> [(word, split, ann, img)] chosen: for each word, up to `per` outlined objects from the train photos and `test` from the val
    photos, each at least `min_side` px on its shorter side, not a crowd, one object a photo per word (the photos differ)"""
    import random
    rng_ = random.Random(seed)
    out_ = []
    for split_, f_, n_ in (("train", "annotations/instances_train2017.json", per), ("test", "annotations/instances_val2017.json", test)):
        d_ = json.load(open(os.path.join(root, f_)))
        cats_ = {c_["id"]: c_["name"] for c_ in d_["categories"]}
        imgs_ = {i_["id"]: i_ for i_ in d_["images"]}
        by_ = {}
        for a_ in d_["annotations"]:
            nm_ = cats_.get(a_["category_id"])
            if nm_ not in WORDS or a_.get("iscrowd"):
                continue
            x_, y_, w_, h_ = a_["bbox"]
            if min(w_, h_) < min_side:
                continue
            by_.setdefault(WORDS[nm_], []).append(a_)
        for w_, anns_ in sorted(by_.items()):
            rng_.shuffle(anns_)
            seen_ = set(); k_ = 0
            for a_ in anns_:
                if a_["image_id"] in seen_:
                    continue
                seen_.add(a_["image_id"]); out_.append((w_, split_, a_, imgs_[a_["image_id"]])); k_ += 1
                if k_ >= n_:
                    break
    return out_


def crop_one(root, px, item, index, licences):
    from PIL import Image
    w_, split_, a_, im_ = item
    d_ = os.path.join(root, "crops", w_)
    p_ = os.path.join(d_, f"{a_['id']}.jpg")
    if not os.path.exists(p_):
        url_ = im_["coco_url"]
        if not re.match(r"^http://images\.cocodataset\.org/(train|val)2017/\d+\.jpg$", url_):
            return
        raw_ = _get(url_)
        img_ = Image.open(io.BytesIO(raw_))
        if img_.format != "JPEG":
            return
        img_ = img_.convert("RGB")
        x_, y_, bw_, bh_ = a_["bbox"]
        s_ = max(bw_, bh_) * 1.25; cx_, cy_ = x_ + bw_ / 2.0, y_ + bh_ / 2.0     # (square, a margin of an eighth each side)
        box_ = (int(round(cx_ - s_ / 2)), int(round(cy_ - s_ / 2)), int(round(cx_ + s_ / 2)), int(round(cy_ + s_ / 2)))
        os.makedirs(d_, exist_ok=True)
        img_.crop(box_).resize((px, px), Image.LANCZOS).save(p_, quality=90)
    with _lock:
        index[str(a_["id"])] = dict(word=w_, split=split_, image=im_["id"], file=f"{w_}/{a_['id']}.jpg", bbox=[round(v_, 1) for v_ in a_["bbox"]],
                                    licence=licences.get(im_.get("license")), flickr=im_.get("flickr_url"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/photos/coco")
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args(argv)
    fetch_annotations(a.root)
    lic_ = {l_["id"]: l_["name"] for l_ in json.load(open(os.path.join(a.root, "annotations/instances_val2017.json")))["licenses"]}
    items_ = plan(a.root, a.per, a.test, a.min_side)
    print(f"{len(items_)} objects to cut ({sum(1 for i_ in items_ if i_[1] == 'train')} lessons, {sum(1 for i_ in items_ if i_[1] == 'test')} tests)", flush=True)
    idx_p_ = os.path.join(a.root, "crops", "index.json")
    os.makedirs(os.path.dirname(idx_p_), exist_ok=True)
    index_ = json.load(open(idx_p_)) if os.path.exists(idx_p_) else {}
    done_ = [0]

    def job(it_):
        try:
            crop_one(a.root, a.px, it_, index_, lic_)
        except Exception as e_:
            print(f"skip {it_[0]} {it_[2]['id']}: {e_}", flush=True)
        done_[0] += 1
        if done_[0] % 500 == 0:
            print(f"{done_[0]} of {len(items_)}", flush=True)
            with _lock:
                json.dump(index_, open(idx_p_, "w"))
    with ThreadPoolExecutor(a.threads) as ex_:
        list(ex_.map(job, items_))
    json.dump(index_, open(idx_p_, "w"))
    from collections import Counter
    c_ = Counter((v_["word"], v_["split"]) for v_ in index_.values())
    print(f"{len(index_)} crops; words {len({w_ for w_, _ in c_})}")


if __name__ == "__main__":
    main()
