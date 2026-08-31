"""Extract WildFake's DALL-E Advanced eval set without downloading the 25.6 GB zip.

DALLE.zip is served over HTTP with Range support, so the 8843 members under
DALLE/Advanced/ can be fetched directly: read the ZIP64 central directory from
the tail, then pull only those members' byte ranges. 2.84 GB instead of 25.59 GB.

Offsets exceed 4 GB, so ZIP64 extra fields are mandatory; the 32-bit central
directory fields read 0xFFFFFFFF.

Held-out eval data. Never train on the output of this script.

Usage: uv run python scripts/extract_dalle_advanced.py [--limit N]
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

URL = "https://www.modelscope.cn/datasets/hy2628982280/WildFake/resolve/master/Images/Diffusion_based/DALLE.zip"
PREFIX = "DALLE/Advanced/"
SPAN_BYTES = 32 * 1024 * 1024
SLACK = 4096


def http_range(start, end, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            return urllib.request.urlopen(req, timeout=300).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry {attempt+1} after {type(exc).__name__}", file=sys.stderr)


def read_central_directory():
    size = int(urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60)
               .headers["Content-Length"])
    tail = http_range(size - 65600, size - 1)
    i = tail.rfind(b"PK\x06\x06")
    if i >= 0:
        cd_size = struct.unpack("<Q", tail[i + 40:i + 48])[0]
        cd_off = struct.unpack("<Q", tail[i + 48:i + 56])[0]
    else:
        j = tail.rfind(b"PK\x05\x06")
        cd_size = struct.unpack("<I", tail[j + 12:j + 16])[0]
        cd_off = struct.unpack("<I", tail[j + 16:j + 20])[0]
    return http_range(cd_off, cd_off + cd_size + 65536), size


def parse_entries(cd):
    entries, p = [], 0
    while True:
        k = cd.find(b"PK\x01\x02", p)
        if k < 0:
            return entries
        nlen, mlen, clen = struct.unpack("<HHH", cd[k + 28:k + 34])
        csize, usize = struct.unpack("<II", cd[k + 20:k + 28])
        lho = struct.unpack("<I", cd[k + 42:k + 46])[0]
        name = cd[k + 46:k + 46 + nlen].decode("utf-8", "replace")
        extra = cd[k + 46 + nlen:k + 46 + nlen + mlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                blk, o = extra[q + 4:q + 4 + hsz], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", blk[o:o + 8])[0]; o += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", blk[o:o + 8])[0]; o += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", blk[o:o + 8])[0]; o += 8
                break
            q += 4 + hsz
        entries.append({"name": name, "method": struct.unpack("<H", cd[k + 10:k + 12])[0],
                        "csize": csize, "usize": usize, "lho": lho})
        p = k + 46 + nlen + mlen + clen


def inflate_member(blob, offset, entry):
    """Parse a local file header at `offset` within `blob` and return the member bytes."""
    if blob[offset:offset + 4] != b"PK\x03\x04":
        return None
    nlen, mlen = struct.unpack("<HH", blob[offset + 26:offset + 30])
    start = offset + 30 + nlen + mlen
    data = blob[start:start + entry["csize"]]
    if len(data) < entry["csize"]:
        return None
    return zlib.decompress(data, -zlib.MAX_WBITS) if entry["method"] == 8 else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/dalle_advanced")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cd, total = read_central_directory()
    members = [e for e in parse_entries(cd)
               if e["name"].startswith(PREFIX) and not e["name"].endswith("/")]
    members.sort(key=lambda e: e["lho"])
    if args.limit:
        members = members[:args.limit]
    print(f"zip {total/1e9:.2f} GB | {len(members)} Advanced members | "
          f"{sum(e['csize'] for e in members)/1e9:.2f} GB to fetch")

    for e in members:
        rel = e["name"][len(PREFIX):]
        e["dest"] = os.path.join(args.out, rel)
    todo = [e for e in members if not os.path.exists(e["dest"])]
    print(f"{len(members) - len(todo)} already present, {len(todo)} to go")

    spans, cur = [], []
    for e in todo:
        if cur and (e["lho"] + e["csize"] + SLACK) - cur[0]["lho"] > SPAN_BYTES:
            spans.append(cur); cur = []
        cur.append(e)
    if cur:
        spans.append(cur)

    written = failed = 0
    for n, span in enumerate(spans, 1):
        lo = span[0]["lho"]
        hi = span[-1]["lho"] + 30 + SLACK + span[-1]["csize"]
        blob = http_range(lo, hi)
        for e in span:
            try:
                data = inflate_member(blob, e["lho"] - lo, e)
                if data is None:
                    solo = http_range(e["lho"], e["lho"] + 30 + SLACK + e["csize"])
                    data = inflate_member(solo, 0, e)
                if data is None:
                    failed += 1
                    continue
                os.makedirs(os.path.dirname(e["dest"]), exist_ok=True)
                with open(e["dest"], "wb") as f:
                    f.write(data)
                written += 1
            except Exception as exc:
                print(f"  FAILED {e['name']}: {type(exc).__name__} {exc}", file=sys.stderr)
                failed += 1
        print(f"[{n}/{len(spans)}] written={written} failed={failed}", flush=True)

    print(f"done: {written} written, {failed} failed -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
