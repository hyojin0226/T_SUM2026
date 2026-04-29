# 데이터 경로를 매번 길게 쓰지 않기 위한 설정 파일
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DATA_DIR = DATA_DIR / "output"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MAP_DIR = OUTPUT_DIR / "maps"

# Analysis settings
TARGET_YEAR = 2021
ADMIN_UNIT = "행정동"