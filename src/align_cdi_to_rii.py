"""Align CDI preprocessing outputs to the 425-dong RII 2021 boundary system.

This script intentionally does not merge on dong name alone. It standardizes
gu + dong names, applies a small administrative-change mapping, aggregates raw
infrastructure counts, recalculates CDI, then replaces CDI administrative codes
with the RII administrative codes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("c:/Tsum2026/T_SUM2026")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CDI_2021_PATH = PROCESSED_DIR / "cdi_by_dong_2021.csv"
CDI_ALL_PATH = PROCESSED_DIR / "cdi_by_dong_2019_2021.csv"
SANGGA_2021_PATH = PROCESSED_DIR / "sangga_infra_by_dong_2021.csv"
RII_2021_PATH = PROCESSED_DIR / "rii_by_dong_2021.csv"

OUT_CDI_2021 = PROCESSED_DIR / "cdi_by_dong_2021_aligned.csv"
OUT_CDI_ALL = PROCESSED_DIR / "cdi_by_dong_2019_2021_aligned.csv"
OUT_UNMATCHED_CDI = TABLE_DIR / "cdi_unmatched_after_alignment.csv"
OUT_UNMATCHED_RII = TABLE_DIR / "rii_unmatched_after_cdi_alignment.csv"
OUT_REVIEW = TABLE_DIR / "cdi_alignment_manual_review.csv"

COUNT_COLS = [
    "convenience_count",
    "grocery_count",
    "pharmacy_count",
    "medical_count",
]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_admin_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize gu/dong names before matching.

    - strip leading/trailing whitespace
    - remove internal whitespace from dong names
    - normalize '.' to '·'
    """
    out = df.copy()
    out["자치구명"] = out["자치구명"].astype(str).str.strip()
    out["행정동명"] = (
        out["행정동명"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "·", regex=False)
    )
    out["dong_key"] = out["자치구명"] + "_" + out["행정동명"]
    return out


def apply_manual_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Map split/merged administrative dongs to the RII 2021 dong system."""
    out = df.copy()
    manual_map = {
        ("강동구", "상일1동"): "상일동",
        ("강동구", "상일2동"): "상일동",
        ("동대문구", "신설동"): "용신동",
        ("동대문구", "용두동"): "용신동",
        # 2022-12-23부터 일원2동이 개포3동으로 명칭 변경됨.
        # 최종 분석 기준은 RII의 2021년 행정동 체계이므로 개포3동을 일원2동으로 되돌립니다.
        ("강남구", "개포3동"): "일원2동",
    }

    out["mapping_note"] = ""
    for (gu, old_dong), new_dong in manual_map.items():
        mask = (out["자치구명"] == gu) & (out["행정동명"] == old_dong)
        out.loc[mask, "mapping_note"] = f"{old_dong}->{new_dong}"
        out.loc[mask, "행정동명"] = new_dong

    out["dong_key"] = out["자치구명"] + "_" + out["행정동명"]
    return out


def minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.max() == values.min():
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / (values.max() - values.min())


def recalculate_cdi(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate CDI after the dong system has been aligned.

    Infrastructure counts are summed first. More infrastructure means lower
    care-desert vulnerability, so CDI is 1 - infra_score.
    """
    out = df.copy()
    for col in COUNT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["total_infra_count"] = out[COUNT_COLS].sum(axis=1)
    out["infra_score"] = out.groupby("기준연도")["total_infra_count"].transform(minmax_scale)
    out["CDI"] = 1 - out["infra_score"]
    out["CDI_rank"] = out.groupby("기준연도")["CDI"].rank(ascending=False, method="min").astype(int)

    parts = []
    for year, part in out.groupby("기준연도", dropna=False):
        part = part.copy()
        if part["CDI"].nunique(dropna=False) <= 1:
            part["CDI_grade"] = 3
        else:
            part["CDI_grade"] = pd.qcut(
                part["CDI"].rank(method="first"),
                q=5,
                labels=[1, 2, 3, 4, 5],
            ).astype(int)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def aggregate_counts_to_target_system(cdi: pd.DataFrame) -> pd.DataFrame:
    """Aggregate original infrastructure count columns after manual mapping."""
    keep_cols = ["기준연도", "자치구명", "행정동명", "dong_key", *COUNT_COLS]
    agg = (
        cdi[keep_cols]
        .groupby(["기준연도", "자치구명", "행정동명", "dong_key"], as_index=False, dropna=False)
        .agg({col: "sum" for col in COUNT_COLS})
    )
    return agg


def align_to_rii_template(cdi_agg: pd.DataFrame, rii_2021: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Left-align CDI to the RII 425-dong template for each target year."""
    rii_template = clean_admin_names(rii_2021[["자치구명", "행정동명", "행정동코드"]].copy())
    rii_template = rii_template.drop_duplicates(["행정동코드", "dong_key"])
    rii_template["_rii_order"] = np.arange(len(rii_template))

    template_parts = []
    for year in years:
        tmp = rii_template.copy()
        tmp["기준연도"] = year
        template_parts.append(tmp)
    template = pd.concat(template_parts, ignore_index=True)

    aligned = template.merge(
        cdi_agg.drop(columns=["자치구명", "행정동명"], errors="ignore"),
        on=["기준연도", "dong_key"],
        how="left",
        indicator=True,
    )

    # Keep RII's official gu/dong/code. CDI's old code is intentionally not used.
    unmatched_rii = aligned[aligned["_merge"] == "left_only"][
        ["기준연도", "자치구명", "행정동명", "행정동코드", "dong_key"]
    ].copy()

    for col in COUNT_COLS:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0)

    aligned = aligned.drop(columns=["_merge"])
    aligned = recalculate_cdi(aligned)
    return aligned, unmatched_rii


def main() -> None:
    cdi_2021 = read_csv(CDI_2021_PATH)
    cdi_all = read_csv(CDI_ALL_PATH)
    sangga_2021 = read_csv(SANGGA_2021_PATH)
    rii_2021 = read_csv(RII_2021_PATH)

    print("=== loaded ===")
    for name, df in [
        ("cdi_2021", cdi_2021),
        ("cdi_all", cdi_all),
        ("sangga_2021", sangga_2021),
        ("rii_2021", rii_2021),
    ]:
        print(name, df.shape)

    cdi_all_name_clean = clean_admin_names(cdi_all)
    cdi_2021_name_clean = clean_admin_names(cdi_2021)
    cdi_all_clean = apply_manual_mapping(cdi_all_name_clean)
    cdi_2021_clean = apply_manual_mapping(cdi_2021_name_clean)
    rii_clean = clean_admin_names(rii_2021)

    # sangga_infra_by_dong_2021.csv는 CDI 2021의 원 인프라 개수와 같은지 확인용으로 불러옵니다.
    # 이 검증은 수동 매핑 전의 원 행정동 체계에서 수행해야 합니다.
    sangga_clean = clean_admin_names(sangga_2021)
    cdi_count_check = cdi_2021_name_clean[["자치구명", "행정동명", *COUNT_COLS]].merge(
        sangga_clean[["자치구명", "행정동명", *COUNT_COLS]],
        on=["자치구명", "행정동명"],
        how="outer",
        suffixes=("_cdi", "_sangga"),
        indicator=True,
    )
    mismatch_count = 0
    for col in COUNT_COLS:
        left = pd.to_numeric(cdi_count_check[f"{col}_cdi"], errors="coerce").fillna(-1)
        right = pd.to_numeric(cdi_count_check[f"{col}_sangga"], errors="coerce").fillna(-1)
        mismatch_count += int((left != right).sum())
    print("sangga_infra_by_dong_2021 count mismatch cells:", mismatch_count)

    # First matching check before manual mapping is already available from
    # cdi_2021_clean's mapping_note. Print and save the remaining mismatches.
    cdi_keys = set(cdi_2021_clean["dong_key"])
    rii_keys = set(rii_clean["dong_key"])
    cdi_unmatched = cdi_2021_clean[~cdi_2021_clean["dong_key"].isin(rii_keys)].copy()
    rii_unmatched = rii_clean[~rii_clean["dong_key"].isin(cdi_keys)].copy()
    review = cdi_2021_clean[cdi_2021_clean["mapping_note"].str.contains("검토", na=False)].copy()

    print("\n=== unmatched after cleaning + manual mapping ===")
    print("CDI not in RII:", cdi_unmatched[["자치구명", "행정동명", "행정동코드", "mapping_note"]].to_string(index=False))
    print("RII not in CDI:", rii_unmatched[["자치구명", "행정동명", "행정동코드"]].to_string(index=False))
    print("\n=== manual review ===")
    print(review[["자치구명", "행정동명", "행정동코드", "mapping_note"]].to_string(index=False))

    cdi_unmatched.to_csv(OUT_UNMATCHED_CDI, index=False, encoding="utf-8-sig")
    rii_unmatched.to_csv(OUT_UNMATCHED_RII, index=False, encoding="utf-8-sig")
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    years = sorted(cdi_all_clean["기준연도"].unique().tolist())
    cdi_agg = aggregate_counts_to_target_system(cdi_all_clean)
    aligned_all, unmatched_rii_all = align_to_rii_template(cdi_agg, rii_2021, years)

    final_cols = [
        "_rii_order",
        "기준연도",
        "자치구명",
        "행정동코드",
        "행정동명",
        *COUNT_COLS,
        "total_infra_count",
        "infra_score",
        "CDI",
        "CDI_rank",
        "CDI_grade",
    ]
    aligned_all = aligned_all[final_cols].sort_values(["기준연도", "_rii_order"]).reset_index(drop=True)
    aligned_2021 = (
        aligned_all[aligned_all["기준연도"] == 2021]
        .copy()
        .sort_values("_rii_order")
        .reset_index(drop=True)
    )
    aligned_all = aligned_all.drop(columns=["_rii_order"])
    aligned_2021 = aligned_2021.drop(columns=["_rii_order"])

    aligned_all.to_csv(OUT_CDI_ALL, index=False, encoding="utf-8-sig")
    aligned_2021.to_csv(OUT_CDI_2021, index=False, encoding="utf-8-sig")

    print("\n=== final validation ===")
    print("aligned_all shape:", aligned_all.shape)
    print("aligned_2021 shape:", aligned_2021.shape)
    print("2021 row count == 425:", len(aligned_2021) == 425)
    print("행정동코드 duplicate rows:", aligned_2021.duplicated(["행정동코드"]).sum())
    print("key null counts:")
    print(aligned_2021[["기준연도", "자치구명", "행정동명", "행정동코드", "CDI", "CDI_grade"]].isna().sum())

    rii_codes = set(rii_2021["행정동코드"].astype(str))
    cdi_codes = set(aligned_2021["행정동코드"].astype(str))
    print("RII/CDI code 100% match:", rii_codes == cdi_codes)
    print("code overlap:", len(rii_codes & cdi_codes), "/", len(rii_codes))
    print("CDI in 0~1:", aligned_2021["CDI"].between(0, 1).all())
    print("CDI_grade in 1~5:", aligned_2021["CDI_grade"].between(1, 5).all())
    print("unmatched RII rows with zero-filled counts:", len(unmatched_rii_all))
    if len(unmatched_rii_all):
        print(unmatched_rii_all.drop_duplicates(["자치구명", "행정동명"]).to_string(index=False))

    print("\nSaved:")
    print(OUT_CDI_2021)
    print(OUT_CDI_ALL)
    print(OUT_UNMATCHED_CDI)
    print(OUT_UNMATCHED_RII)
    print(OUT_REVIEW)


if __name__ == "__main__":
    main()
