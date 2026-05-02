"""Create report-ready map and chart PNGs for the final 2021 CCI outputs."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE_DIR = Path("c:/Tsum2026/T_SUM2026")
DATA_OUTPUT_DIR = BASE_DIR / "data" / "output"
MAP_DIR = BASE_DIR / "outputs" / "maps"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
MAP_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CSV_PATH = DATA_OUTPUT_DIR / "final_crisis_index_by_dong_2021.csv"
FINAL_GPKG_PATH = DATA_OUTPUT_DIR / "final_crisis_index_by_dong_2021.gpkg"
TOP_RISK_CSV_PATH = DATA_OUTPUT_DIR / "top_risk_dongs_2021.csv"

SOURCE_NOTE = "자료: 서울시 빅데이터캠퍼스 분석 결과(2021)"
DPI = 350
FIGSIZE_MAP = (9, 10)
FIGSIZE_BAR = (9, 6)

GRADE_COLORS = ["#fff7bc", "#fec44f", "#fe9929", "#ec7014", "#b10026"]
GRADE_CMAP = ListedColormap(GRADE_COLORS)
GRADE_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], GRADE_CMAP.N)

RISK_TYPE_ORDER = [
    "복합위기 최고위험지역",
    "사회고립+인프라취약형",
    "사회고립+수해취약형",
    "인프라+수해취약형",
    "사회고립 단일취약형",
    "인프라 단일취약형",
    "수해 단일취약형",
    "상대적 저위험지역",
]

RISK_TYPE_COLORS = {
    "복합위기 최고위험지역": "#b10026",
    "사회고립+인프라취약형": "#e31a1c",
    "사회고립+수해취약형": "#fb6a4a",
    "인프라+수해취약형": "#fdae6b",
    "사회고립 단일취약형": "#6a51a3",
    "인프라 단일취약형": "#3182bd",
    "수해 단일취약형": "#31a354",
    "상대적 저위험지역": "#d9d9d9",
}


def setup_korean_font() -> None:
    """Configure a Korean font that works well on Windows/Jupyter."""
    candidates = [
        "Malgun Gothic",
        "맑은 고딕",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    mpl.rcParams["font.family"] = selected
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["savefig.facecolor"] = "white"
    print(f"사용 폰트: {selected}")


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def add_report_title(ax, title: str, subtitle: str = "2021년 서울시 행정동 기준") -> None:
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    ax.text(
        0.5,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
    )


def add_source_note(fig, note: str = SOURCE_NOTE) -> None:
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9, color="#666666")


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"저장 완료: {output_path}")


def make_grade_from_value(gdf: gpd.GeoDataFrame, value_col: str, grade_col: str) -> gpd.GeoDataFrame:
    """Create a stable 1~5 quantile grade if the grade column is missing."""
    out = gdf.copy()
    if grade_col in out.columns and out[grade_col].notna().any():
        out[grade_col] = pd.to_numeric(out[grade_col], errors="coerce").astype("Int64")
        return out
    out[grade_col] = pd.qcut(
        pd.to_numeric(out[value_col], errors="coerce").rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    print(f"[정보] {grade_col} 컬럼이 없어 {value_col} 기준 분위수 등급을 생성했습니다.")
    return out


def load_data() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final geometry data and supporting CSVs."""
    gdf = gpd.read_file(FINAL_GPKG_PATH)
    final_df = pd.read_csv(FINAL_CSV_PATH, encoding="utf-8-sig")
    top_risk_df = pd.read_csv(TOP_RISK_CSV_PATH, encoding="utf-8-sig")

    for df in [gdf, final_df, top_risk_df]:
        if "행정동코드" in df.columns:
            df["행정동코드"] = normalize_code(df["행정동코드"])

    for value_col, grade_col in [
        ("IVI", "IVI_grade"),
        ("CDI", "CDI_grade"),
        ("RII", "RII_grade"),
        ("CCI", "CCI_grade"),
    ]:
        gdf = make_grade_from_value(gdf, value_col, grade_col)

    return gdf, final_df, top_risk_df


def basic_data_check(gdf: gpd.GeoDataFrame) -> None:
    """Print checks that should be reviewed before map generation."""
    required_cols = [
        "IVI",
        "CDI",
        "RII",
        "CCI",
        "IVI_grade",
        "CDI_grade",
        "RII_grade",
        "CCI_grade",
        "triple_high_flag",
        "risk_type",
        "자치구명",
        "행정동명",
        "행정동코드",
        "geometry",
    ]
    missing_cols = [col for col in required_cols if col not in gdf.columns]

    print("[기본 데이터 점검]")
    print("gdf 행 수:", len(gdf))
    print("CRS:", gdf.crs)
    print("geometry 결측치 수:", int(gdf.geometry.isna().sum()))
    print("geometry empty 수:", int(gdf.geometry.is_empty.sum()))
    print("geometry 유효 개수:", int(gdf.geometry.is_valid.sum()), "/", len(gdf))
    print("필수 컬럼 누락:", missing_cols)
    for col in ["IVI_grade", "CDI_grade", "RII_grade", "CCI_grade"]:
        print(f"{col} 값:", sorted(pd.Series(gdf[col]).dropna().astype(int).unique().tolist()))
    print("triple_high_flag 값:")
    print(gdf["triple_high_flag"].value_counts(dropna=False).sort_index())
    print("risk_type 고유값:")
    print(gdf["risk_type"].value_counts().reindex(RISK_TYPE_ORDER).dropna().astype(int))


def plot_grade_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    cmap=GRADE_CMAP,
) -> None:
    """Create a 1~5 grade choropleth map."""
    plot_gdf = gdf.copy()
    plot_gdf[column] = pd.to_numeric(plot_gdf[column], errors="coerce").astype("Int64")

    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    plot_gdf.plot(
        column=column,
        cmap=cmap,
        norm=GRADE_NORM,
        linewidth=0.25,
        edgecolor="#666666",
        ax=ax,
        missing_kwds={"color": "#f0f0f0", "edgecolor": "#999999", "label": "결측"},
    )
    ax.set_axis_off()
    add_report_title(ax, title)

    legend_labels = [
        "1등급 낮음",
        "2등급",
        "3등급",
        "4등급",
        "5등급 높음",
    ]
    handles = [Patch(facecolor=GRADE_COLORS[i], edgecolor="#666666", label=legend_labels[i]) for i in range(5)]
    ax.legend(
        handles=handles,
        title="위험 등급",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=11,
        title_fontsize=12,
    )
    add_source_note(fig)
    save_figure(fig, output_path)


def plot_value_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    cmap: str = "YlOrRd",
) -> None:
    """Create a continuous value map for optional appendix figures."""
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    gdf.plot(
        column=column,
        cmap=cmap,
        linewidth=0.25,
        edgecolor="#666666",
        ax=ax,
        legend=True,
        legend_kwds={"shrink": 0.55, "label": f"{column} 값"},
    )
    ax.set_axis_off()
    add_report_title(ax, title)
    add_source_note(fig)
    save_figure(fig, output_path)


def plot_binary_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
) -> None:
    """Create a binary highlight map for triple_high_flag."""
    plot_gdf = gdf.copy()
    plot_gdf[column] = pd.to_numeric(plot_gdf[column], errors="coerce").fillna(0).astype(int)

    colors = {0: "#e6e6e6", 1: "#b10026"}
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    for value, color in colors.items():
        subset = plot_gdf[plot_gdf[column] == value]
        if len(subset):
            subset.plot(color=color, linewidth=0.25, edgecolor="#666666", ax=ax)
    ax.set_axis_off()
    add_report_title(ax, title)

    handles = [
        Patch(facecolor="#b10026", edgecolor="#666666", label="복합위기 최고위험지역"),
        Patch(facecolor="#e6e6e6", edgecolor="#666666", label="기타 지역"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=11,
    )
    add_source_note(fig)
    save_figure(fig, output_path)


def plot_categorical_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    category_order: list[str],
    color_dict: dict[str, str],
) -> None:
    """Create a categorical risk type map with a fixed legend order."""
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    for category in category_order:
        subset = gdf[gdf[column] == category]
        if len(subset) == 0:
            continue
        subset.plot(
            color=color_dict.get(category, "#cccccc"),
            linewidth=0.25,
            edgecolor="#666666",
            ax=ax,
        )
    ax.set_axis_off()
    add_report_title(ax, title)

    handles = [
        Patch(facecolor=color_dict[category], edgecolor="#666666", label=category)
        for category in category_order
        if category in set(gdf[column])
    ]
    ax.legend(
        handles=handles,
        title="위험 유형",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9,
        title_fontsize=11,
    )
    add_source_note(fig)
    save_figure(fig, output_path)


def plot_top10_bar(df: pd.DataFrame, value_col: str, title: str, output_path: Path) -> None:
    """Create a top 10 horizontal bar chart."""
    plot_df = df.sort_values(value_col, ascending=False).head(10).copy()
    plot_df["label"] = plot_df["자치구명"] + " " + plot_df["행정동명"]
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    bars = ax.barh(plot_df["label"], plot_df[value_col], color="#d7301f")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel(f"{value_col} 값", fontsize=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=10)
    add_source_note(fig)
    save_figure(fig, output_path)


def plot_risk_type_distribution(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    counts = gdf["risk_type"].value_counts().reindex(RISK_TYPE_ORDER).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [RISK_TYPE_COLORS[category] for category in counts.index]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_title("복합위기 유형별 행정동 수", fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel("행정동 수", fontsize=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=10)
    add_source_note(fig)
    save_figure(fig, output_path)


def create_all_maps_and_figures(create_value_maps: bool = True, create_bar_charts: bool = True) -> None:
    setup_korean_font()
    gdf, final_df, top_risk_df = load_data()
    basic_data_check(gdf)

    grade_maps = [
        ("IVI_grade", "행정동별 사회적 고립 취약도(IVI)", MAP_DIR / "ivi_grade_map_2021.png"),
        ("CDI_grade", "행정동별 생활 인프라 사막 지수(CDI)", MAP_DIR / "cdi_grade_map_2021.png"),
        ("RII_grade", "행정동별 강우·수해 취약도(RII)", MAP_DIR / "rii_grade_map_2021.png"),
        ("CCI_grade", "행정동별 복합위기지수(CCI)", MAP_DIR / "cci_grade_map_2021.png"),
    ]
    for column, title, output_path in grade_maps:
        plot_grade_map(gdf, column, title, output_path)

    plot_binary_map(
        gdf,
        "triple_high_flag",
        "복합위기 최고위험지역 분포",
        MAP_DIR / "triple_high_map_2021.png",
    )
    plot_categorical_map(
        gdf,
        "risk_type",
        "행정동별 복합위기 유형 분포",
        MAP_DIR / "risk_type_map_2021.png",
        RISK_TYPE_ORDER,
        RISK_TYPE_COLORS,
    )

    if create_value_maps:
        value_maps = [
            ("IVI", "행정동별 사회적 고립 취약도(IVI) 값", MAP_DIR / "ivi_value_map_2021.png"),
            ("CDI", "행정동별 생활 인프라 사막 지수(CDI) 값", MAP_DIR / "cdi_value_map_2021.png"),
            ("RII", "행정동별 강우·수해 취약도(RII) 값", MAP_DIR / "rii_value_map_2021.png"),
            ("CCI", "행정동별 복합위기지수(CCI) 값", MAP_DIR / "cci_value_map_2021.png"),
        ]
        for column, title, output_path in value_maps:
            plot_value_map(gdf, column, title, output_path)

    if create_bar_charts:
        plot_top10_bar(final_df, "IVI", "IVI 상위 10개 행정동", FIGURE_DIR / "ivi_top10_bar.png")
        plot_top10_bar(final_df, "CDI", "CDI 상위 10개 행정동", FIGURE_DIR / "cdi_top10_bar.png")
        plot_top10_bar(final_df, "RII", "RII 상위 10개 행정동", FIGURE_DIR / "rii_top10_bar.png")
        plot_top10_bar(final_df, "CCI", "CCI 상위 10개 행정동", FIGURE_DIR / "cci_top10_bar.png")
        plot_risk_type_distribution(gdf, FIGURE_DIR / "risk_type_distribution.png")

    print("\n필수 지도 저장 확인")
    required_outputs = [
        MAP_DIR / "ivi_grade_map_2021.png",
        MAP_DIR / "cdi_grade_map_2021.png",
        MAP_DIR / "rii_grade_map_2021.png",
        MAP_DIR / "cci_grade_map_2021.png",
        MAP_DIR / "triple_high_map_2021.png",
        MAP_DIR / "risk_type_map_2021.png",
    ]
    for path in required_outputs:
        print(path, path.exists())


if __name__ == "__main__":
    create_all_maps_and_figures(create_value_maps=True, create_bar_charts=True)
