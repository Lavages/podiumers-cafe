import os
import polars as pl
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

# Path Matrix Optimization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TSV_PATH = os.path.join(DATA_DIR, "WCA_export_results.tsv")
PARQUET_PATH = os.path.join(DATA_DIR, "WCA_export_results.parquet")

EVENT_NAMES = {
    "222": "2x2x2 Cube", "333": "3x3x3 Cube", "444": "4x4x4 Cube", 
    "555": "5x5x5 Cube", "666": "6x6x6 Cube", "777": "7x7x7 Cube",
    "333bf": "3x3x3 Blindfolded", "333fm": "3x3x3 Fewest Moves", 
    "333oh": "3x3x3 One-Handed", "clock": "Clock", 
    "minx": "Megaminx", "pyram": "Pyraminx", "skewb": "Skewb", 
    "sq1": "Square-1", "444bf": "4x4x4 Blindfolded", 
    "555bf": "5x5x5 Blindfolded", "333mbf": "3x3x3 Multi-Blind", "333mbo": "3x3x3 Multi-Blind Oldstyle",
    "magic": "Magic", "mmagic": "Master Magic", "333ft": "3x3x3 With Feet"
}

def verify_and_compile_dataset():
    """
    Validates structural integrity of application assets. Evaluates binary Parquet availability.
    Aggressively groups and sums records during compilation to smash file size to < 10MB,
    strictly omitting DNF result markings.
    """
    if os.path.exists(PARQUET_PATH):
        return True

    if not os.path.exists(TSV_PATH):
        print(f"[CRITICAL ERROR] Target path source assets missing at: {TSV_PATH}")
        return False

    print(f"[COMPILING ENGINE] Aggressively pre-aggregating database matrix...")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 1. Scan raw data lazily
        lazy_tsv = pl.scan_csv(TSV_PATH, separator="\t", has_header=True, infer_schema_length=0)
        
        # 2. Filter for podium positions, final rounds, AND explicitly exclude DNF values ("-1")
        # In the WCA database schema, a single/average DNF status is represented as -1.
        filtered = lazy_tsv.filter(
            (pl.col("pos").is_in(["1", "2", "3"])) &
            (pl.col("round_type_id").is_in(["f", "c"])) &
            (pl.col("best") != "-1") &       # Exclude best single DNF values
            (pl.col("average") != "-1")      # Exclude average DNF values
        )
        
        # 3. Collapse the entire dataset by grouping names/IDs and positions immediately
        compact_df = (
            filtered.group_by(["person_id", "person_name", "event_id", "pos"])
            .agg(pl.len().alias("count"))
            .select([
                pl.col("person_id").cast(pl.Categorical),
                pl.col("person_name").cast(pl.Categorical),
                pl.col("event_id").cast(pl.Categorical),
                pl.col("pos").cast(pl.Int8),
                pl.col("count").cast(pl.Int32)
            ])
            .collect()
        )
        
        # 4. Save with high-density zstd compression
        compact_df.write_parquet(PARQUET_PATH, compression="zstd", compression_level=12)
        
        actual_size = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
        print(f"[SYSTEM SUCCESS] Micro-optimized Parquet constructed at: {PARQUET_PATH}")
        print(f"-> Final optimized deployment size: {actual_size:.2f} MB")
        return True
    except Exception as e:
        print(f"[COMPILATION FAILURE] Runtime serialization halted: {str(e)}")
        return False

def get_competitor_podiums(search_query: str):
    """
    Queries highly compressed pre-aggregated datasets instantly with safe string casting.
    """
    search_query = search_query.strip()
    if not search_query:
        return None, None, None

    # Load high-speed schema mapping
    lazy_df = pl.scan_parquet(PARQUET_PATH)

    # Filter evaluation
    matched_lazy = lazy_df.filter(
        (pl.col("person_id").cast(pl.Utf8) == search_query) | 
        (pl.col("person_name").cast(pl.Utf8).str.to_lowercase().str.contains(search_query.lower()))
    )

    results_df = matched_lazy.collect()

    if results_df.is_empty():
        return None, None, f"No verified podium records discovered matching input: '{search_query}'"

    # Convert values to strings safely for tracking and template presentation formatting
    results_df = results_df.with_columns([
        pl.col("person_name").cast(pl.Utf8),
        pl.col("person_id").cast(pl.Utf8),
        pl.col("event_id").cast(pl.Utf8)
    ])

    # Resolve accurate competitor identities safely from string distributions
    competitor_name = results_df["person_name"].value_counts().sort("count", descending=True)["person_name"][0]
    competitor_id = results_df["person_id"].value_counts().sort("count", descending=True)["person_id"][0]

    # Sum the pre-computed 'count' metrics
    pivoted = (
        results_df.pivot(on="pos", index="event_id", values="count", aggregate_function="sum")
        .fill_null(0)
    )

    # Confirm column alignments
    for pos_col in [1, 2, 3]:
        if str(pos_col) not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).alias(str(pos_col)))

    pivoted = pivoted.rename({"1": "gold", "2": "silver", "3": "bronze"})
    pivoted = pivoted.with_columns((pl.col("gold") + pl.col("silver") + pl.col("bronze")).alias("total")).sort("total", descending=True)

    # Compute explicit overall metrics safely
    totals = {
        "gold": int(pivoted["gold"].sum()),
        "silver": int(pivoted["silver"].sum()),
        "bronze": int(pivoted["bronze"].sum()),
        "overall": int(pivoted["total"].sum())
    }

    summary_list = []
    for row in pivoted.to_dicts():
        summary_list.append({
            "event_name": EVENT_NAMES.get(row["event_id"], row["event_id"]),
            "gold": row["gold"], "silver": row["silver"], "bronze": row["bronze"], "total": row["total"]
        })

    profile = {"name": competitor_name, "wca_id": competitor_id}
    return profile, totals, summary_list

@app.route("/", methods=["GET"])
def index():
    query = request.args.get("query", "")
    if not query:
        return render_template("index.html")

    if not verify_and_compile_dataset():
        return render_template("index.html", error="Local records engine uninitialized. Ensure database assets are available.")

    try:
        profile, totals, data = get_competitor_podiums(query)
        if profile is None:
            return render_template("index.html", error=data)
        return render_template("index.html", profile=profile, totals=totals, data=data)
    except Exception as e:
        return render_template("index.html", error=f"Data system execution anomaly: {str(e)}")

if __name__ == "__main__":
    verify_and_compile_dataset()
    app.run(debug=True, port=5000)