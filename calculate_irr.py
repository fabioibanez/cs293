import itertools
import math
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import cohen_kappa_score


DATA_PATH = "IRR Calculation - Construct 2.csv"
EXPERTS = ["Simran", "Febie", "Fabio"]
CONSTRUCTS = ["Offering mathematical help", "Successful Uptake", "Asking for More Information"]


def _pairwise_expert_columns(construct: str) -> Dict[str, str]:
	return {expert: f"{expert} - {construct}" for expert in EXPERTS}


def _clean_binary(series: pd.Series) -> pd.Series:
	cleaned = (
		series.astype(str)
		.str.strip()
		.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
	)
	return pd.to_numeric(cleaned, errors="coerce")


def _percent_agreement(a: pd.Series, b: pd.Series) -> float:
	mask = a.notna() & b.notna()
	if mask.sum() == 0:
		return math.nan
	return (a[mask] == b[mask]).mean()


def _cohens_kappa(a: pd.Series, b: pd.Series) -> float:
	mask = a.notna() & b.notna()
	if mask.sum() == 0:
		return math.nan
	return cohen_kappa_score(a[mask], b[mask])


def _pairwise_metrics(df: pd.DataFrame, construct: str) -> pd.DataFrame:
	columns = _pairwise_expert_columns(construct)
	metrics: List[Tuple[str, str, float, float, float]] = []

	for expert_a, expert_b in itertools.combinations(EXPERTS, 2):
		col_a = columns[expert_a]
		col_b = columns[expert_b]
		series_a = _clean_binary(df[col_a])
		series_b = _clean_binary(df[col_b])

		agreement = _percent_agreement(series_a, series_b)
		kappa = _cohens_kappa(series_a, series_b)
		corr = series_a.corr(series_b)
		metrics.append((expert_a, expert_b, agreement, kappa, corr))

	return pd.DataFrame(
		metrics,
		columns=[
			"Expert A",
			"Expert B",
			"Percent Agreement",
			"Cohen's Kappa",
			"Correlation",
		],
	)


def _fleiss_kappa(df: pd.DataFrame, construct: str) -> float:
	columns = _pairwise_expert_columns(construct)
	rating_cols = [columns[expert] for expert in EXPERTS]
	ratings = df[rating_cols].apply(_clean_binary)
	ratings = ratings.dropna(how="any")
	if ratings.empty:
		return math.nan

	n = len(EXPERTS)
	# counts per item for categories 0 and 1
	count_1 = ratings.sum(axis=1)
	count_0 = n - count_1

	p_i = (count_0 * (count_0 - 1) + count_1 * (count_1 - 1)) / (n * (n - 1))
	p_bar = p_i.mean()

	p_0 = count_0.sum() / (len(ratings) * n)
	p_1 = count_1.sum() / (len(ratings) * n)
	p_e = p_0 ** 2 + p_1 ** 2

	if math.isclose(1 - p_e, 0.0):
		return math.nan
	return (p_bar - p_e) / (1 - p_e)


def main() -> None:
	df = pd.read_csv(DATA_PATH)

	for construct in CONSTRUCTS:
		metrics_df = _pairwise_metrics(df, construct)
		print(f"\n=== {construct} ===")
		print(metrics_df.to_string(index=False))
		fleiss = _fleiss_kappa(df, construct)
		print(f"Fleiss' Kappa (all four raters): {fleiss:.4f}")


if __name__ == "__main__":
	main()