# fast_color_eval.py
# 최적화판: LUT + 멀티프로세싱 + I/O 최소화
# 기능: PNG(무손실) 화질 best/worst, JPEG(quality 고정) 용량 min/max, CSV 집계
# 폴더: dataset/<style>/* → outputs_color_eval/<style>/
import os, io, glob, re, math, csv
import numpy as np
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor, as_completed
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================== 사용자 설정 ==================
DATASET_ROOT = "dataset"
OUT_ROOT     = "outputs_color_eval_fast"

MODE         = "YCbCr"   # 'RGB' or 'YCbCr'  (YCbCr=Y만 처리)
LEVELS       = 16        # 양자화 단계
SAVE_REC_IMAGES = True   # True면 결과 이미지 저장

# JPEG(용량 비교용)
JPEG_QUALITY = 85
SUBSAMPLING  = 2         # 4:2:0

# 함수 그리드
INCLUDE_SQRT     = True
GRID_POWER_GAMMA = [0.4, 0.5, 0.8, 1.2]
GRID_RATIONAL_K  = [64, 128, 256]
GRID_LOG_ALPHA   = [4.0, 8.0, 16.0]
GRID_EXP_BETA    = [2.0, 4.0, 8.0]
GRID_SIG_A       = [6.0, 10.0, 12.0]
GRID_SIG_C       = [0.40, 0.50, 0.60]

# 병렬
MAX_WORKERS = None  # None=CPU 코어 수
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
# =================================================

def safe_tag(s, maxlen=80): return re.sub(r"[^A-Za-z0-9._-]+","_",str(s))[:maxlen].strip("_")

def normalize_to_png(path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out  = os.path.join(cache_dir, base + ".png")
    if not os.path.exists(out):
        img = Image.open(path)
        if img.mode not in ("RGB","RGBA"): img = img.convert("RGB")
        img.save(out, format="PNG", optimize=True, compress_level=9)
    return out

def mse(a,b): a=a.astype(np.float32); b=b.astype(np.float32); return float(np.mean((a-b)**2))
def psnr(a,b):
    m=mse(a,b)
    return float("inf") if m==0 else 20*math.log10(255.0)-10*math.log10(m)

# -------- LUT 생성 --------
def make_lut_pair(method, params, levels):
    x = np.arange(256, dtype=np.float32)
    _u = lambda z: np.clip(z,0,255)/255.0
    _y = lambda u: 255.0*np.clip(u,0,1)

    if method=="sqrt":
        f  = _y(np.sqrt(_u(x)))
        fi = _y((_u(x))**2)
    elif method=="power":
        g  = float(params["gamma"])
        f  = _y((_u(x))**g)
        fi = _y((_u(x))**(1.0/g if g!=0 else 1.0))
    elif method=="rational":
        k  = float(params["k"])
        x2 = np.clip(x,0,255)
        f  = 255.0*x2/(x2+k)
        y  = np.clip(x,0,254.9999)
        fi = (k*y)/(255.0-y)
    elif method=="log":
        a  = float(params["alpha"])
        f  = _y(np.log1p(a*_u(x))/math.log1p(a))
        fi = _y(np.expm1(_u(x)*math.log1p(a))/a)
    elif method=="exp":
        b  = float(params["beta"]); den = math.expm1(b)
        f  = _y(np.expm1(b*_u(x))/(den if den!=0 else 1))
        fi = _y(np.log1p(_u(x)*den)/(b if b!=0 else 1))
    elif method=="sigmoid":
        a=float(params["a"]); c=float(params["c"])
        S=lambda u: 1.0/(1.0+np.exp(-a*(u-c)))
        u=_u(x); s0,s1=S(0),S(1)
        f  = _y((S(u)-s0)/(s1-s0+1e-12))
        t  = _u(x); s = s0 + t*(s1-s0); s=np.clip(s,1e-12,1-1e-12)
        fi = _y(c-(1.0/a)*np.log(1.0/s-1.0))
    else:
        raise ValueError("method")

    step = 255.0/(levels-1)
    return f.astype(np.float32), fi.astype(np.float32), np.float32(step)

def apply_chain_u8(arr_u8, f_lut, inv_lut, step):
    y   = f_lut[arr_u8]                     # x -> f(x)
    yq  = np.round(y/step)*step             # 균일 양자화
    idx = np.clip(np.rint(yq), 0, 255).astype(np.uint8)
    xr  = inv_lut[idx]                      # inv
    return np.clip(np.rint(xr),0,255).astype(np.uint8)

def reconstruct_color(pil_img, f_lut, inv_lut, step, mode):
    if mode=="RGB":
        arr = np.array(pil_img.convert("RGB"), dtype=np.uint8, copy=True)  # writeable
        out = np.empty_like(arr)
        for c in range(3):
            out[...,c] = apply_chain_u8(arr[...,c], f_lut, inv_lut, step)
        return Image.fromarray(out, "RGB")

    elif mode=="YCbCr":
        ycb = np.array(pil_img.convert("YCbCr"), dtype=np.uint8, copy=True)  # writeable
        Yrec = apply_chain_u8(ycb[...,0], f_lut, inv_lut, step)
        ycb[...,0] = Yrec
        return Image.fromarray(ycb, "YCbCr").convert("RGB")

    else:
        raise ValueError("MODE")

def jpeg_size_bytes(pil_img, quality=JPEG_QUALITY, subsampling=SUBSAMPLING):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG",
                                quality=int(quality),
                                subsampling=subsampling,
                                optimize=True,
                                progressive=True)
    return len(buf.getvalue())

# -------- 파이프라인 정의(부모 프로세스에서 1회) --------
def build_pipelines():
    items=[]
    if INCLUDE_SQRT:
        items.append(("sqrt", {"gamma":0.5}))
    for g in GRID_POWER_GAMMA:
        items.append(("power", {"gamma":g}))
    for k in GRID_RATIONAL_K:
        items.append(("rational", {"k":k}))
    for a in GRID_LOG_ALPHA:
        items.append(("log", {"alpha":a}))
    for b in GRID_EXP_BETA:
        items.append(("exp", {"beta":b}))
    for a in GRID_SIG_A:
        for c in GRID_SIG_C:
            items.append(("sigmoid", {"a":a,"c":c}))
    return items

# -------- 워커: 이미지 1장 처리(모든 조합) --------
def process_one_image(image_path_png, style, outdir, mode, levels, save_imgs, jpeg_quality):
    img = Image.open(image_path_png)
    orig = np.asarray(img.convert("RGB"), dtype=np.float32)
    base = os.path.splitext(os.path.basename(image_path_png))[0]
    base_tag = safe_tag(base, 80)

    if save_imgs:
        os.makedirs(os.path.join(outdir,"best_rec_png"),  exist_ok=True)
        os.makedirs(os.path.join(outdir,"worst_rec_png"), exist_ok=True)
        os.makedirs(os.path.join(outdir,"minsize_jpeg"), exist_ok=True)
        os.makedirs(os.path.join(outdir,"maxsize_jpeg"), exist_ok=True)

    # 결과 누적
    rows_png = []   # [method, params_str, levels, mse, psnr]
    rows_jpg = []   # [method, params_str, levels, sizeB]

    for method, params in build_pipelines():
        f_lut, inv_lut, step = make_lut_pair(method, params, levels)
        rec = reconstruct_color(img, f_lut, inv_lut, step, mode)

        # PNG 화질
        rec_rgb = np.asarray(rec.convert("RGB"), dtype=np.float32)
        p = psnr(orig, rec_rgb); m = mse(orig, rec_rgb)
        rows_png.append([method, str(params), levels, m, p])

        # JPEG 용량
        sizeB = jpeg_size_bytes(rec, jpeg_quality)
        rows_jpg.append([method, str(params), levels, sizeB])

    # 정렬 및 선택
    rows_png.sort(key=lambda r:(-r[4], r[3]))  # PSNR desc
    rows_jpg.sort(key=lambda r:(r[3],))        # size asc

    best_png  = rows_png[0]
    worst_png = rows_png[-1]
    min_jpg   = rows_jpg[0]
    max_jpg   = rows_jpg[-1]

    # 선택 저장
    def save_png(method, params_str, subdir):
        if not save_imgs: return
        f_lut, inv_lut, step = make_lut_pair(method, eval(params_str), levels)
        rec = reconstruct_color(img, f_lut, inv_lut, step, mode)
        pm_tag = safe_tag(params_str)
        name = f"{base_tag}__{method}_{pm_tag}_{mode}_L{levels}.png"
        rec.save(os.path.join(outdir, subdir, name), format="PNG", optimize=True, compress_level=9)

    def save_jpg(method, params_str, subdir, label):
        if not save_imgs: return
        f_lut, inv_lut, step = make_lut_pair(method, eval(params_str), levels)
        rec = reconstruct_color(img, f_lut, inv_lut, step, mode)
        pm_tag = safe_tag(params_str)
        name = f"{base_tag}__{method}_{pm_tag}_{mode}_L{levels}_{label}.jpg"
        rec.convert("RGB").save(os.path.join(outdir, subdir, name),
                                format="JPEG", quality=jpeg_quality,
                                subsampling=SUBSAMPLING, optimize=True, progressive=True)

    save_png(best_png[0],  best_png[1],  "best_rec_png")
    save_png(worst_png[0], worst_png[1], "worst_rec_png")
    save_jpg(min_jpg[0],  min_jpg[1],  "minsize_jpeg", "min")
    save_jpg(max_jpg[0],  max_jpg[1],  "maxsize_jpeg", "max")

    # CSV용 행 반환
    png_row  = [style, base, best_png[0], best_png[1], best_png[2], best_png[3], best_png[4],
                worst_png[0], worst_png[1], worst_png[3], worst_png[4]]
    jpeg_row = [style, base, min_jpg[0], min_jpg[1], min_jpg[2], min_jpg[3],
                max_jpg[0], max_jpg[1], max_jpg[2], max_jpg[3]]
    return png_row, jpeg_row

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    cache_dir = os.path.join(OUT_ROOT, "_normalized_png"); os.makedirs(cache_dir, exist_ok=True)

    styles = [d for d in sorted(os.listdir(DATASET_ROOT))
              if os.path.isdir(os.path.join(DATASET_ROOT, d))]

    tasks = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for style in styles:
            style_dir = os.path.join(DATASET_ROOT, style)
            outdir    = os.path.join(OUT_ROOT, style); os.makedirs(outdir, exist_ok=True)

            img_paths=[]
            for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"):
                img_paths += glob.glob(os.path.join(style_dir, ext))
            img_paths = sorted(img_paths)
            if not img_paths: continue

            # 입력 PNG 정규화(부모 프로세스에서 선행, I/O 중복 방지)
            norm_paths = [normalize_to_png(p, cache_dir) for p in img_paths]
            for pn in norm_paths:
                tasks.append(ex.submit(process_one_image, pn, style, outdir, MODE, LEVELS, SAVE_REC_IMAGES, JPEG_QUALITY))

        # 수집
        rows_png_all  = []
        rows_jpeg_all = []
        done = 0
        for fut in as_completed(tasks):
            png_row, jpeg_row = fut.result()
            rows_png_all.append(png_row)
            rows_jpeg_all.append(jpeg_row)
            done += 1
            if done % 20 == 0: print(f"processed {done}/{len(tasks)} images")

    # 스타일별 CSV와 전체 CSV 생성
    # 메모리 내 행을 스타일별로 그룹화
    by_style_png  = {}
    by_style_jpeg = {}
    for r in rows_png_all:
        by_style_png.setdefault(r[0], []).append(r)
    for r in rows_jpeg_all:
        by_style_jpeg.setdefault(r[0], []).append(r)

    # 스타일별 CSV
    for style in styles:
        outdir = os.path.join(OUT_ROOT, style)
        if style in by_style_png:
            with open(os.path.join(outdir, f"{style}_best_worst_png.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(
                    ["style","image","best_method","best_params","levels","best_mse","best_psnr",
                     "worst_method","worst_params","worst_mse","worst_psnr"])
                for row in by_style_png[style]: w.writerow(row)
        if style in by_style_jpeg:
            with open(os.path.join(outdir, f"{style}_min_max_jpeg.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(
                    ["style","image","minsize_method","minsize_params","levels","min_size_bytes",
                     "maxsize_method","maxsize_params","levels","max_size_bytes"])
                for row in by_style_jpeg[style]: w.writerow(row)

    # 전체 CSV
    with open(os.path.join(OUT_ROOT, "ALL_best_worst_png.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(
            ["style","image","best_method","best_params","levels","best_mse","best_psnr",
             "worst_method","worst_params","worst_mse","worst_psnr"])
        for r in rows_png_all: w.writerow(r)

    with open(os.path.join(OUT_ROOT, "ALL_min_max_jpeg.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(
            ["style","image","minsize_method","minsize_params","levels","min_size_bytes",
             "maxsize_method","maxsize_params","levels","max_size_bytes"])
        for r in rows_jpeg_all: w.writerow(r)

    print("Done:", os.path.abspath(OUT_ROOT))

if __name__ == "__main__":
    main()
