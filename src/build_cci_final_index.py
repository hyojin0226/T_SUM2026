"""Build final 2021 CCI by merging IVI, CDI, and RII on RII dong codes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("c:/Tsum2026/T_SUM2026")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IVI_PATH = PROCESSED_DIR / "ivi_social_isolation_2021.csv"
CDI_PATH = PROCESSED_DIR / "cdi_by_dong_2021_v2.csv"   # 재설계 CDI (고령자 기준 정규화 + 로그변환)
RII_PATH = PROCESSED_DIR / "rii_by_dong_2021_v2.csv"       # 공간 이질성 변수 추가 버전 (불투수면+DEM)
RII_GPKG_PATH = PROCESSED_DIR / "rii_by_dong_2021_v2.gpkg"

OUT_FINAL_CSV = OUTPUT_DIR / "final_crisis_index_by_dong_2021.csv"
OUT_TOP_CSV = OUTPUT_DIR / "top_risk_dongs_2021.csv"
OUT_FINAL_GPKG = OUTPUT_DIR / "final_crisis_index_by_dong_2021.gpkg"

# 가중치 근거 (AHP 기반 우선순위 반영):
#   IVI 0.40 — 프로젝트 핵심 대상(고령 1인가구 사회적 고립)이 주제 전면에 있음
#   RII 0.35 — "비가 오면"이라는 강우 맥락이 제목에 명시되어 있어 두 번째 우선순위
#   CDI 0.25 — 강우 시 이동 불가로 인프라 부족이 위기를 증폭하는 보조 요인
WEIGHTS = {
    "IVI": 0.40,
    "CDI": 0.25,
    "RII": 0.35,
}

IVI_CANDIDATES = ["IVI", "ivi", "social_isolation_index", "isolation_index"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_admin_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "자치구명" in out.columns:
        out["자치구명"] = out["자치구명"].astype(str).str.strip()
    if "행정동명" in out.columns:
        out["행정동명"] = (
            out["행정동명"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str.replace(".", "·", regex=False)
        )
    if {"자치구명", "행정동명"}.issubset(out.columns):
        out["dong_key"] = out["자치구명"] + "_" + out["행정동명"]
    return out


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.max() == values.min():
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / (values.max() - values.min())


def add_rank_grade(df: pd.DataFrame, score_col: str, rank_col: str, grade_col: str) -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = out[score_col].rank(ascending=False, method="min").astype(int)
    if out[score_col].nunique(dropna=False) <= 1:
        out[grade_col] = 3
    else:
        try:
            out[grade_col] = pd.qcut(
                out[score_col].rank(method="first"),
                q=5,
                labels=[1, 2, 3, 4, 5],
            ).astype(int)
        except ValueError:
            out[grade_col] = pd.qcut(
                out[score_col].rank(method="first"),
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates="drop",
            )
    return out


def apply_ivi_manual_mapping(ivi: pd.DataFrame) -> pd.DataFrame:
    out = ivi.copy()
    manual_map = {
        ("강동구", "상일1동"): "상일동",
        ("강동구", "상일2동"): "상일동",
    }
    out["mapping_note"] = ""
    for (gu, old_dong), new_dong in manual_map.items():
        mask = (out["자치구명"] == gu) & (out["행정동명"] == old_dong)
        out.loc[mask, "mapping_note"] = f"{old_dong}->{new_dong}"
        out.loc[mask, "행정동명"] = new_dong
    out["dong_key"] = out["자치구명"] + "_" + out["행정동명"]
    return out


def prepare_ivi_to_rii_codes(ivi_raw: pd.DataFrame, rii: pd.DataFrame) -> pd.DataFrame:
    """Attach RII dong codes to IVI and recalculate IVI on the 425-dong system.

    The source IVI file has no dong code. We only use gu+dong to map IVI rows to
    the RII code system, then the final merge uses 행정동코드.
    """
    ivi = ivi_raw.rename(columns={"year": "기준연도", "gu": "자치구명", "dong": "행정동명"}).copy()
    ivi = clean_admin_names(ivi)
    ivi = apply_ivi_manual_mapping(ivi)

    count_cols = ["total_population", "elderly_population", "elderly_alone"]
    missing_counts = [col for col in count_cols if col not in ivi.columns]
    if missing_counts:
        print("IVI 원 카운트 컬럼이 없어 기존 IVI 컬럼 후보를 사용합니다.")
        print("사용 가능한 IVI 후보:", [col for col in IVI_CANDIDATES if col in ivi.columns])
        ivi_col = next((col for col in IVI_CANDIDATES if col in ivi.columns), None)
        if ivi_col is None:
            raise ValueError(f"IVI 지수 컬럼을 찾지 못했습니다. 후보: {IVI_CANDIDATES}")
        ivi_agg = ivi[["기준연도", "자치구명", "행정동명", "dong_key", ivi_col]].rename(columns={ivi_col: "IVI"})
    else:
        for col in count_cols:
            ivi[col] = pd.to_numeric(ivi[col], errors="coerce").fillna(0)
        ivi_agg = (
            ivi.groupby(["기준연도", "자치구명", "행정동명", "dong_key"], as_index=False)
            .agg(
                total_population=("total_population", "sum"),
                elderly_population=("elderly_population", "sum"),
                elderly_alone=("elderly_alone", "sum"),
            )
        )
        ivi_agg["elderly_ratio"] = np.where(
            ivi_agg["total_population"] > 0,
            ivi_agg["elderly_population"] / ivi_agg["total_population"],
            0,
        )
        ivi_agg["elderly_alone_ratio"] = np.where(
            ivi_agg["elderly_population"] > 0,
            ivi_agg["elderly_alone"] / ivi_agg["elderly_population"],
            0,
        )
        ivi_agg["elderly_ratio_score"] = minmax_scale(ivi_agg["elderly_ratio"])
        ivi_agg["elderly_alone_score"] = minmax_scale(ivi_agg["elderly_alone_ratio"])
        ivi_agg["IVI"] = ivi_agg[["elderly_ratio_score", "elderly_alone_score"]].mean(axis=1)

    rii_template = clean_admin_names(rii[["기준연도", "자치구명", "행정동명", "행정동코드"]].copy())
    rii_template["행정동코드"] = normalize_code(rii_template["행정동코드"])
    ivi_aligned = rii_template.merge(
        ivi_agg.drop(columns=["자치구명", "행정동명"], errors="ignore"),
        on=["기준연도", "dong_key"],
        how="left",
        indicator=True,
    )

    missing = ivi_aligned[ivi_aligned["_merge"] == "left_only"][
        ["기준연도", "자치구명", "행정동명", "행정동코드"]
    ]
    if len(missing):
        print("[경고] RII 기준에 매칭되지 않은 IVI 행정동:")
        print(missing.to_string(index=False))

    ivi_aligned = ivi_aligned.drop(columns=["_merge"])
    ivi_aligned = add_rank_grade(ivi_aligned, "IVI", "IVI_rank_recalc", "IVI_grade_recalc")
    return ivi_aligned


def validate_before_merge(name: str, df: pd.DataFrame, index_col: str) -> None:
    print(f"\n=== {name} 병합 전 검증 ===")
    print("행 수:", len(df))
    print("행정동코드 dtype:", df["행정동코드"].dtype)
    print("행정동코드 중복 수:", df.duplicated(["행정동코드"]).sum())
    print(f"{index_col} 결측치 수:", df[index_col].isna().sum())


def classify_risk_type(row: pd.Series) -> str:
    flags = (row["IVI_top30_flag"], row["CDI_top30_flag"], row["RII_top30_flag"])
    if flags == (1, 1, 1):
        return "복합위기 최고위험지역"
    if flags == (1, 1, 0):
        return "사회고립+인프라취약형"
    if flags == (1, 0, 1):
        return "사회고립+수해취약형"
    if flags == (0, 1, 1):
        return "인프라+수해취약형"
    if flags == (1, 0, 0):
        return "사회고립 단일취약형"
    if flags == (0, 1, 0):
        return "인프라 단일취약형"
    if flags == (0, 0, 1):
        return "수해 단일취약형"
    return "상대적 저위험지역"


def main() -> None:
    ivi_raw = read_csv(IVI_PATH)
    cdi = read_csv(CDI_PATH)
    rii = read_csv(RII_PATH)

    print("=== 데이터 불러오기 ===")
    print("IVI:", ivi_raw.shape)
    print("CDI:", cdi.shape)
    print("RII:", rii.shape)

    cdi = clean_admin_names(cdi)
    rii = clean_admin_names(rii)
    cdi["행정동코드"] = normalize_code(cdi["행정동코드"])
    rii["행정동코드"] = normalize_code(rii["행정동코드"])
    ivi = prepare_ivi_to_rii_codes(ivi_raw, rii)
    ivi["행정동코드"] = normalize_code(ivi["행정동코드"])

    validate_before_merge("IVI", ivi, "IVI")
    validate_before_merge("CDI", cdi, "CDI")
    validate_before_merge("RII", rii, "RII")

    print("\n=== 행정동코드 집합 일치 여부 ===")
    code_sets = {
        "IVI": set(ivi["행정동코드"]),
        "CDI": set(cdi["행정동코드"]),
        "RII": set(rii["행정동코드"]),
    }
    print("IVI == RII:", code_sets["IVI"] == code_sets["RII"])
    print("CDI == RII:", code_sets["CDI"] == code_sets["RII"])
    print("IVI-RII 차이:", sorted(code_sets["IVI"] ^ code_sets["RII"])[:20])
    print("CDI-RII 차이:", sorted(code_sets["CDI"] ^ code_sets["RII"])[:20])

    base_cols = ["기준연도", "자치구명", "행정동명", "행정동코드", "RII", "RII_rank", "RII_grade"]
    df = rii[base_cols].copy()
    df = df.merge(
        ivi[["행정동코드", "IVI", "IVI_rank_recalc", "IVI_grade_recalc"]],
        on="행정동코드",
        how="left",
    )
    df = df.merge(
        cdi[["행정동코드", "CDI", "CDI_rank", "CDI_grade"]],
        on="행정동코드",
        how="left",
    )

    print("\n=== 병합 후 검증 ===")
    print("행 수:", len(df))
    print("IVI 누락 행정동:")
    print(df[df["IVI"].isna()][["자치구명", "행정동명", "행정동코드"]].to_string(index=False))
    print("CDI 누락 행정동:")
    print(df[df["CDI"].isna()][["자치구명", "행정동명", "행정동코드"]].to_string(index=False))

    df["CCI"] = (
        df["IVI"] * WEIGHTS["IVI"]
        + df["CDI"] * WEIGHTS["CDI"]
        + df["RII"] * WEIGHTS["RII"]
    )
    df = add_rank_grade(df, "CCI", "CCI_rank", "CCI_grade")

    for index_col in ["IVI", "CDI", "RII"]:
        threshold = df[index_col].quantile(0.70)
        df[f"{index_col}_top30_flag"] = (df[index_col] >= threshold).astype(int)
        print(f"{index_col} top30 threshold:", threshold)

    df["triple_high_flag"] = (
        (df["IVI_top30_flag"] == 1)
        & (df["CDI_top30_flag"] == 1)
        & (df["RII_top30_flag"] == 1)
    ).astype(int)
    df["risk_type"] = df.apply(classify_risk_type, axis=1)

    final_cols = [
        "기준연도",
        "자치구명",
        "행정동명",
        "행정동코드",
        "IVI",
        "CDI",
        "RII",
        "CCI",
        "CCI_rank",
        "CCI_grade",
        "IVI_top30_flag",
        "CDI_top30_flag",
        "RII_top30_flag",
        "triple_high_flag",
        "risk_type",
    ]
    final_df = df[final_cols].copy()
    top_risk = final_df[final_df["triple_high_flag"] == 1].sort_values("CCI_rank").copy()

    final_df.to_csv(OUT_FINAL_CSV, index=False, encoding="utf-8-sig")
    top_risk.to_csv(OUT_TOP_CSV, index=False, encoding="utf-8-sig")

    if RII_GPKG_PATH.exists():
        import geopandas as gpd

        map_gdf = gpd.read_file(RII_GPKG_PATH)
        map_gdf["행정동코드"] = normalize_code(map_gdf["행정동코드"])
        final_gdf = map_gdf[["행정동코드", "geometry"]].merge(final_df, on="행정동코드", how="left")
        final_gdf = gpd.GeoDataFrame(final_gdf, geometry="geometry", crs=map_gdf.crs)
        print("GPKG 행 수:", len(final_gdf))
        print("GPKG CRS:", final_gdf.crs)
        final_gdf.to_file(OUT_FINAL_GPKG, layer="final_crisis_index_by_dong_2021", driver="GPKG")
    else:
        print(f"[주의] RII GPKG가 없어 지도용 GPKG 생성을 건너뜁니다: {RII_GPKG_PATH}")

    print("\n[최종 검증]")
    print("최종 행 수:", len(final_df))
    print("행정동코드 중복 개수:", final_df.duplicated(["행정동코드"]).sum())
    print("IVI 결측치 수:", final_df["IVI"].isna().sum())
    print("CDI 결측치 수:", final_df["CDI"].isna().sum())
    print("RII 결측치 수:", final_df["RII"].isna().sum())
    print("CCI 결측치 수:", final_df["CCI"].isna().sum())
    print("CCI 범위 0~1 여부:", final_df["CCI"].between(0, 1).all())
    print("CCI_grade 범위 1~5 여부:", final_df["CCI_grade"].between(1, 5).all())
    print("triple_high_flag 개수:", int(final_df["triple_high_flag"].sum()))
    print("risk_type별 행정동 수:")
    print(final_df["risk_type"].value_counts().to_string())
    print("상위 10개 CCI 행정동:")
    print(
        final_df.sort_values("CCI_rank")[
            ["자치구명", "행정동명", "IVI", "CDI", "RII", "CCI", "CCI_rank", "CCI_grade", "risk_type"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUT_FINAL_CSV)
    print(OUT_TOP_CSV)
    if OUT_FINAL_GPKG.exists():
        print(OUT_FINAL_GPKG)


if __name__ == "__main__":
    main()
