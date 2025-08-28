# 빠른 컬러 평가 (main.py)

`main.py`는 컬러 이미지에 대한 비선형 밝기 변환을 대량 평가하는 스크립트입니다. 8비트 LUT(룩업 테이블)와 멀티프로세싱을 사용해 빠르게 처리하며, 각 이미지에 대해 PNG 기준 최고/최저 화질과 JPEG 기준 최소/최대 용량을 요약해 CSV로 저장합니다.

## 개요
- 대상 파일: `main.py` (코드 헤더에 fast_color_eval 표기)
- 핵심 아이디어: 변환 도메인에서 균일 양자화를 수행하기 위해, 방법/파라미터 조합별로 8비트 LUT를 미리 만들어 전방/역변환을 빠르게 적용 → 원본과 PSNR/MSE 계산 → JPEG 용량도 추정
- 병렬 처리: `ProcessPoolExecutor`로 여러 이미지를 동시에 처리

## 요구 사항
- Python 3.10+ (권장)
- 패키지: `numpy`, `Pillow`

설치:
```bash
pip install numpy pillow
```

## 데이터셋 구조
아래와 같이 스타일별 하위 폴더에 이미지를 배치하세요.
```
dataset/
  <style1>/
    img1.png
    img2.jpg
  <style2>/
    ...
```
일반적인 이미지 포맷을 지원하며, 내부적으로 한 번 `PNG`로 정규화하여 중복 디코딩을 줄입니다.

## 설정 (main.py 상단 상수)
- `DATASET_ROOT`: 입력 루트(기본 `dataset`)
- `OUT_ROOT`: 출력 루트(기본 `outputs_color_eval_fast`)
- `MODE`: `"RGB"` 또는 `"YCbCr"` (YCbCr는 Y 채널만 처리)
- `LEVELS`: 양자화 단계 수 (예: 16)
- `SAVE_REC_IMAGES`: 대표 복원 PNG/JPEG 파일 저장 여부
- `JPEG_QUALITY`: JPEG 품질(기본 85)
- `SUBSAMPLING`: 2는 4:2:0 (Pillow 옵션)
- 방법/파라미터 그리드(탐색 범위):
  - `INCLUDE_SQRT`
  - `GRID_POWER_GAMMA`, `GRID_RATIONAL_K`, `GRID_LOG_ALPHA`, `GRID_EXP_BETA`
  - `GRID_SIG_A`, `GRID_SIG_C`
- 병렬: `MAX_WORKERS=None`이면 CPU 코어 수 사용. BLAS 스레드는 `OMP_NUM_THREADS`, `MKL_NUM_THREADS`로 제한

## 동작 방식 (요약)
1) 입력 이미지를 `OUT_ROOT/_normalized_png/`에 PNG로 1회 정규화 (I/O 최적화)
2) 각 이미지에 대해 모든 방법/파라미터 조합을 수행:
   - `make_lut_pair`로 전방/역방향 LUT 생성
   - `reconstruct_color`로 RGB 전체 또는 Y 채널만 복원
   - 원본 대비 `MSE/PSNR` 계산
   - 메모리 내 JPEG 인코딩으로 크기(바이트) 추정
3) 이미지별로 PNG 기준 최고/최저 화질, JPEG 기준 최소/최대 용량 선택
4) 스타일별/전체 CSV로 집계 저장

## 실행 방법
프로젝트 루트에서:
```bash
python main.py
```
Windows PowerShell:
```powershell
python .\main.py
```

## 출력물
기본 출력 루트: `outputs_color_eval_fast`
- `_normalized_png/`: 정규화된 입력 PNG 캐시
- 스타일별 CSV
  - `<style>_best_worst_png.csv`
    - 컬럼: `style,image,best_method,best_params,levels,best_mse,best_psnr,`\
            `worst_method,worst_params,worst_mse,worst_psnr`
  - `<style>_min_max_jpeg.csv`
    - 컬럼: `style,image,minsize_method,minsize_params,levels,min_size_bytes,`\
            `maxsize_method,maxsize_params,levels,max_size_bytes`
- 전체 CSV
  - `ALL_best_worst_png.csv`
  - `ALL_min_max_jpeg.csv`
- `SAVE_REC_IMAGES=True`일 때 이미지 폴더
  - `best_rec_png/`, `worst_rec_png/`, `minsize_jpeg/`, `maxsize_jpeg/`

## 팁
- `YCbCr` 모드는 보통 더 빠르며 시각적 왜곡이 민감한 Y(휘도)에 집중합니다.
- `LEVELS`가 클수록 양자화 오류는 줄지만, 압축 효과(계단화)는 약해집니다.
- 탐색 시간을 줄이려면 각 `GRID_*` 리스트 크기를 조절하세요.
- 대규모 데이터셋에서는 `MAX_WORKERS`를 적절히 조절하고 BLAS 스레드를 제한하세요.

## 문제 해결
- 손상/잘린 이미지 경고: `ImageFile.LOAD_TRUNCATED_IMAGES=True`로 일부 완화됩니다.
- I/O 병목: 정규화 PNG 캐시로 동일 파일의 중복 디코딩을 피합니다.
- CPU 사용량 과다: `MAX_WORKERS`를 축소하거나, 상단의 `OMP_NUM_THREADS`, `MKL_NUM_THREADS` 값을 줄이세요.
- 경로/문자 인코딩: Windows에서 출력 경로는 ASCII 위주를 권장하거나 UTF-8 대응 쉘을 사용하세요. 