"""Download the four raw datasets used in the paper into data/raw/.

  fp-landscape    Poelwijk, Socolich & Ranganathan, Nat. Commun. 10:4213 (2019),
                  Supplementary Data 5 (41467_2019_12130_MOESM8_ESM.xlsx). Springer
                  returns 403 without a browser User-Agent AND a Referer header, so
                  both are sent. Saved as data/raw/fp_MOESM8_ESM.xlsx.
  tags-math-sx    Benson et al., PNAS 2018 simplicial datasets. Hosted on Google Drive
  NDC-substances  and linked from https://www.cs.cornell.edu/~arb/data/<name>/.
                  Each is a .tar.gz that is extracted in place under data/raw/.
  ACM.mat         ACM bibliographic dataset as redistributed by the HAN repository
                  (Wang et al., WWW 2019). Saved as data/raw/acm/ACM.mat.

Failures are reported and skipped, never substituted. Files already present and
non-trivially sized are left alone, so the script is safe to re-run.

Run: python scripts/00_download.py            # everything
     python scripts/00_download.py fp-landscape acm
"""
import json
import os
import re
import ssl
import sys
import tarfile
import urllib.parse
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(REPO, "data", "raw")
os.makedirs(RAW, exist_ok=True)

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    CTX = ssl.create_default_context()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Google Drive ids, scraped from https://www.cs.cornell.edu/~arb/data/<name>/index.html
GDRIVE = {
    "tags-math-sx":   "1eDevpF6EZs19rLouNpiKGLIlFOLUfKKG",
    "NDC-substances": "1mGOg0DMh46J2zQdimSXMde1pKNtfAdh8",
}

# Direct downloads: (url, destination relative to data/raw/, extra headers)
DIRECT = {
    "fp-landscape": (
        "https://static-content.springer.com/esm/"
        "art%3A10.1038%2Fs41467-019-12130-8/MediaObjects/"
        "41467_2019_12130_MOESM8_ESM.xlsx",
        "fp_MOESM8_ESM.xlsx",
        {"Referer": "https://www.nature.com/articles/s41467-019-12130-8"},
    ),
    "acm": (
        "https://github.com/Jhy1993/HAN/raw/master/data/acm/ACM.mat",
        os.path.join("acm", "ACM.mat"),
        {},
    ),
}

OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(), urllib.request.HTTPSHandler(context=CTX))
OPENER.addheaders = [("User-Agent", UA)]


def _get(url, headers=None, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return OPENER.open(req, timeout=timeout)


def _have(path, min_bytes=1000):
    return os.path.exists(path) and os.path.getsize(path) > min_bytes


def gdrive(fid, dest):
    """Download from Drive, handling the large-file confirmation interstitial."""
    if _have(dest):
        print(f"[have] {os.path.basename(dest)}")
        return dest
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download"
    body = _get(url).read()
    if b"<html" in body[:400].lower():
        txt = body.decode("utf8", "ignore")
        action = re.search(r'action="([^"]+)"', txt)
        fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', txt))
        if not action:
            raise RuntimeError("drive: no confirmation form found")
        u = action.group(1).replace("&amp;", "&") + "?" + urllib.parse.urlencode(fields)
        body = _get(u).read()
    if b"<html" in body[:400].lower():
        raise RuntimeError("drive: still HTML after confirmation")
    open(dest, "wb").write(body)
    print(f"[got ] {os.path.basename(dest)}  {len(body) / 1e6:.1f} MB")
    return dest


def main(which=None):
    log = []
    for name, fid in GDRIVE.items():
        if which and name not in which:
            continue
        dest = os.path.join(RAW, f"{name}.tar.gz")
        try:
            gdrive(fid, dest)
            with tarfile.open(dest) as t:
                t.extractall(RAW)
            log.append([name, "ok", os.path.getsize(dest)])
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            log.append([name, f"FAIL {type(e).__name__}: {e}", 0])
    for name, (url, rel, headers) in DIRECT.items():
        if which and name not in which:
            continue
        dest = os.path.join(RAW, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            if _have(dest):
                print(f"[have] {rel}")
            else:
                b = _get(url, headers).read()
                open(dest, "wb").write(b)
                print(f"[got ] {rel}  {len(b) / 1e6:.1f} MB")
            log.append([name, "ok", os.path.getsize(dest)])
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            log.append([name, f"FAIL {type(e).__name__}: {e}", 0])
    json.dump(log, open(os.path.join(RAW, "download_log.json"), "w"), indent=1)
    print("\nwrote data/raw/download_log.json")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
