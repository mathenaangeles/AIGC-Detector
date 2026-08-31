"""Frozen CLIP ViT-L/14 vision tower with a trainable attention probe.

The tower never trains. It is constructed in eval mode, its parameters have
requires_grad=False, forward runs under no_grad, and `train()` is overridden so
that putting the enclosing model into training mode cannot silently flip the
backbone's LayerNorms back on. Only the probe carries gradient.

Because the tower is frozen, its output for a given crop is a constant. Running
it once and caching the patch tokens turns head training from a ViT-L/14 forward
pass per step into a memmap read, which is the difference between an epoch in
hours and an epoch in seconds.

    STORAGE. 256 patch tokens x 1024 dims x 2 bytes (fp16) = 512 KiB per crop.
    At crops_per_image=4 that is 2 MiB per image: ~33 GB for 16k images. The
    CLS token is stored alongside at 2 KiB. Budget the disk before caching a
    full split; scripts/cache_features.py prints the estimate and --limit
    exists to try it small first.

Caching fixes the crops. CropDataset re-draws crop coordinates every epoch
(rng keyed on epoch), which is good augmentation and incompatible with a cache
-- a cached crop set is drawn once, keyed on the image path and the crop index,
and reused. The trade is deliberate: `crop_boxes` is epoch-independent so the
key is stable, and crops_per_image becomes the whole of the crop augmentation
rather than a per-epoch sample.

The cache also records bias_match settings in meta.json. Features extracted
from unmatched crops carry the container confound that shortcut.py measures,
and reusing such a cache under `bias_match: true` would silently reintroduce
it, so a mismatched cache is refused rather than reused.
"""

import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..data import bias_match, random_crop, reflect_pad_to

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

META_NAME = "meta.json"
INDEX_NAME = "index.json"
SHARD_FMT = "tokens_{:05d}.npy"
CLS_FMT = "cls_{:05d}.npy"


def resolve_device(name=None):
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def crop_boxes(width, height, size, n, seed, key):
    """`n` deterministic crop boxes for one image, independent of epoch.

    Keyed on the image path so adding or reordering images never shifts another
    image's crops, which would invalidate its cache entries.
    """
    width, height = max(width, size), max(height, size)
    stream = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([int(seed), stream])
    boxes = []
    for _ in range(int(n)):
        left = int(rng.integers(0, width - size + 1))
        top = int(rng.integers(0, height - size + 1))
        boxes.append((left, top))
    return boxes


def cache_key(rel_path, box, size):
    return f"{rel_path}@{box[0]},{box[1]},{size}"


def load_crop(path, box, size, do_match=False, quality=90):
    """The exact crop a cache key names. Mirrors CropDataset's pixel path."""
    with Image.open(path) as img:
        img.load()
        img = reflect_pad_to(img.convert("RGB"), size)
        crop = img.crop((box[0], box[1], box[0] + size, box[1] + size))
    if do_match:
        crop = bias_match(crop, quality)
    return crop


def to_pixels(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class FrozenCLIP(nn.Module):
    """The vision tower, and nothing trainable.

    Emits the 256 post-LayerNorm patch tokens (width 1024 for ViT-L/14) and the
    CLS token. The contrastive projection head is not used: it was trained to
    align with text, and the probe wants the spatial tokens.
    """

    def __init__(self, arch="ViT-L-14-quickgelu", pretrained="openai", device=None,
                 cache_dir=None):
        super().__init__()
        import open_clip

        # OpenAI's weights were trained with QuickGELU. open_clip >=2.24 builds
        # plain GELU for the bare "ViT-L-14" name and only warns, which leaves
        # the tower running the wrong activation on the right weights: patch
        # tokens come out at cosine 0.85 to the correct ones. Silent, and it
        # would look like a weak probe rather than a broken backbone.
        if pretrained == "openai" and "quickgelu" not in arch.lower():
            raise ValueError(
                f"{arch} with pretrained='openai' silently uses plain GELU; "
                f"use '{arch}-quickgelu' instead"
            )

        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, cache_dir=cache_dir
        )
        self.visual = model.visual
        self.visual.output_tokens = True
        self.visual.eval()
        self.visual.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

        self.width = int(self.visual.ln_post.normalized_shape[0])
        self.patch_size = int(self.visual.conv1.kernel_size[0])
        if device is not None:
            self.to(device)

    def train(self, mode=True):
        """No-op on the backbone. A frozen tower in train mode is a silent bug."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixels):
        """pixels: (B, 3, H, W) in [0, 1]. Returns (cls, tokens)."""
        x = (pixels.to(self.mean.device) - self.mean) / self.std
        cls, tokens = self.visual(x)
        return cls.float(), tokens.float()

    def n_tokens(self, crop_size):
        return (crop_size // self.patch_size) ** 2


class AttentionProbe(nn.Module):
    """Learned-query attention pooling over patch tokens, then 2 logits.

    One query, `heads` heads. The query is a raw parameter rather than a
    projection of anything, so there is no q_proj to pay for -- the usual MHA
    in_proj of 3*D*D would be 3.1M on its own at D=1024, over budget before the
    classifier exists.
    """

    def __init__(self, width=1024, dim=512, heads=8, n_classes=2, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError(f"probe_dim {dim} must divide probe_heads {heads}")
        self.dim, self.heads, self.head_dim = int(dim), int(heads), int(dim) // int(heads)

        self.norm_in = nn.LayerNorm(width)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.k_proj = nn.Linear(width, dim)
        self.v_proj = nn.Linear(width, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, int(n_classes))
        )
        nn.init.normal_(self.query, std=dim ** -0.5)

    def _split(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens, return_attention=False):
        """tokens: (B, N, width). Returns (B, n_classes)."""
        tokens = self.norm_in(tokens)
        q = self._split(self.query.expand(tokens.shape[0], -1, -1))
        k, v = self._split(self.k_proj(tokens)), self._split(self.v_proj(tokens))

        if return_attention:
            attn = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)
            pooled = attn @ v
        else:
            attn = None
            pooled = F.scaled_dot_product_attention(q, k, v)

        pooled = pooled.transpose(1, 2).reshape(tokens.shape[0], -1)
        pooled = self.norm_out(self.out_proj(pooled))
        logits = self.classifier(self.dropout(pooled))
        return (logits, attn.squeeze(2)) if return_attention else logits


class ClipProbe(nn.Module):
    """Frozen tower plus probe. Trains from cached tokens or raw pixels."""

    def __init__(self, backbone=None, probe=None, **probe_kwargs):
        super().__init__()
        self.backbone = backbone
        self.probe = probe or AttentionProbe(**probe_kwargs)

    def forward(self, pixels):
        if self.backbone is None:
            raise RuntimeError("no backbone attached; call forward_tokens on cached features")
        _, tokens = self.backbone(pixels)
        return self.probe(tokens)

    def forward_tokens(self, tokens):
        return self.probe(tokens)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())


class FeatureCache:
    """Sharded fp16 token cache: `tokens_00000.npy` + a key -> (shard, row) index.

    Not one file per crop. A full split is ~64k crops; that many 512 KiB files
    costs an inode each and turns every training epoch into 64k random opens,
    which on a cluster filesystem is slower than the backbone it replaces.
    Shards are memmapped, so a read is a page fault and nothing more.
    """

    def __init__(self, root, shard_size=512):
        self.root = str(root)
        self.shard_size = int(shard_size)
        self.index = {}
        self.meta = {}
        self._shards = {}
        self._cls_shards = {}
        self._buffer, self._cls_buffer, self._keys = [], [], []
        self._next_shard = 0
        if os.path.exists(os.path.join(self.root, INDEX_NAME)):
            self.load()

    # -- reading ---------------------------------------------------------

    def load(self):
        with open(os.path.join(self.root, INDEX_NAME)) as f:
            self.index = json.load(f)
        with open(os.path.join(self.root, META_NAME)) as f:
            self.meta = json.load(f)
        self._next_shard = 1 + max((v[0] for v in self.index.values()), default=-1)
        return self

    def __contains__(self, key):
        return key in self.index

    def __len__(self):
        return len(self.index)

    def _shard(self, shard_id, store, fmt):
        if shard_id not in store:
            path = os.path.join(self.root, fmt.format(shard_id))
            store[shard_id] = np.load(path, mmap_mode="r")
        return store[shard_id]

    def tokens(self, key, dtype=np.float32):
        shard_id, row = self.index[key]
        return np.asarray(self._shard(shard_id, self._shards, SHARD_FMT)[row], dtype=dtype)

    def cls(self, key, dtype=np.float32):
        shard_id, row = self.index[key]
        return np.asarray(self._shard(shard_id, self._cls_shards, CLS_FMT)[row], dtype=dtype)

    def check_compatible(self, meta):
        """Refuse a cache built under different settings rather than reuse it."""
        if not self.meta:
            return
        mismatched = {k: (self.meta.get(k), v) for k, v in meta.items() if self.meta.get(k) != v}
        if mismatched:
            raise ValueError(
                f"cache at {self.root} was built with different settings: "
                + ", ".join(f"{k} cached={a!r} requested={b!r}" for k, (a, b) in mismatched.items())
                + " -- delete it or point --out elsewhere"
            )

    # -- writing ---------------------------------------------------------

    def open_for_write(self, meta):
        self.check_compatible(meta)
        os.makedirs(self.root, exist_ok=True)
        self.meta = dict(meta)
        return self

    def add(self, key, tokens, cls_token):
        self._keys.append(key)
        self._buffer.append(np.asarray(tokens, dtype=np.float16))
        self._cls_buffer.append(np.asarray(cls_token, dtype=np.float16))
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        shard_id = self._next_shard
        np.save(os.path.join(self.root, SHARD_FMT.format(shard_id)), np.stack(self._buffer))
        np.save(os.path.join(self.root, CLS_FMT.format(shard_id)), np.stack(self._cls_buffer))
        for row, key in enumerate(self._keys):
            self.index[key] = [shard_id, row]
        self._next_shard += 1
        self._buffer, self._cls_buffer, self._keys = [], [], []
        self.save()

    def save(self):
        with open(os.path.join(self.root, INDEX_NAME), "w") as f:
            json.dump(self.index, f)
        with open(os.path.join(self.root, META_NAME), "w") as f:
            json.dump(self.meta, f, indent=2)


def cache_meta(cfg):
    """The settings a cache is only valid under."""
    return {
        "arch": str(cfg.model.clip.arch),
        "pretrained": str(cfg.model.clip.pretrained),
        "crop_size": int(cfg.data.crop_size),
        "crops_per_image": int(cfg.data.crops_per_image),
        "bias_match": bool(cfg.data.bias_match),
        "match_quality": int(cfg.data.match_quality),
        "seed": int(cfg.seed),
        "dtype": "float16",
    }


def cache_dir(cfg):
    clip = cfg.model.clip
    tag = f"{clip.arch}_{clip.pretrained}_c{int(cfg.data.crop_size)}"
    if bool(cfg.data.bias_match):
        tag += f"_q{int(cfg.data.match_quality)}"
    return os.path.join(str(cfg.paths.cache), "clip_tokens", tag)


def enumerate_crops(rows, cfg):
    """Every (row, box, key) the cache should hold for these images."""
    size = int(cfg.data.crop_size)
    n = int(cfg.data.crops_per_image)
    seed = int(cfg.seed)
    out = []
    for row in rows:
        width, height = int(row.get("width") or size), int(row.get("height") or size)
        for box in crop_boxes(width, height, size, n, seed, row["path"]):
            out.append((row, box, cache_key(row["path"], box, size)))
    return out


class CachedTokenDataset(Dataset):
    """Cached patch tokens and a label. No backbone, no image decode."""

    def __init__(self, rows, cfg, cache=None):
        if any(r.get("split") == "eval" for r in rows):
            raise ValueError("eval rows reached CachedTokenDataset; data/eval/ is held out")
        self.cache = cache or FeatureCache(cache_dir(cfg)).load()
        self.cache.check_compatible(cache_meta(cfg))
        self.items = [(key, int(row["label"] == 1))
                      for row, _, key in enumerate_crops(rows, cfg) if key in self.cache]
        missing = len(enumerate_crops(rows, cfg)) - len(self.items)
        if missing:
            raise ValueError(f"{missing} crops are not cached; run scripts/cache_features.py")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, label = self.items[index]
        return torch.from_numpy(self.cache.tokens(key)), torch.tensor(label, dtype=torch.long)


def build_probe(cfg):
    """The trainable half, and a hard check against the parameter budget."""
    clip = cfg.model.clip
    probe = AttentionProbe(
        width=1024 if str(clip.arch).startswith("ViT-L") else 768,
        dim=int(clip.probe_dim), heads=int(clip.probe_heads),
    )
    n = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    if n >= 2_000_000:
        raise ValueError(f"probe has {n:,} trainable params, budget is under 2M")
    return probe


def build(cfg, device=None, with_backbone=True):
    backbone = None
    if with_backbone:
        clip = cfg.model.clip
        backbone = FrozenCLIP(str(clip.arch), str(clip.pretrained), device=resolve_device(device))
    model = ClipProbe(backbone=backbone, probe=build_probe(cfg))
    if device is not None:
        model.probe.to(resolve_device(device))
    return model
