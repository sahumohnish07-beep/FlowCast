"""Generate and save key Exploratory Data Analysis (EDA) visualizations for FlowCast.

Outputs:
    - docs/images/traffic_by_hour.png
    - docs/images/traffic_by_weekday.png
    - docs/images/traffic_by_weather.png
    - docs/images/traffic_by_holiday.png
    - docs/images/correlation_heatmap.png
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_all_plots(
    data_path: str | Path = "data/processed/full.parquet",
    output_dir: str | Path = "docs/images",
) -> None:
    """Generate and export high-resolution EDA charts."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed data from {data_path}...")
    df = pd.read_parquet(data_path)

    # 1. Traffic by Hour (Diurnal curve with weekend comparison)
    print("Generating plot 1: traffic_by_hour.png...")
    plt.figure(figsize=(11, 5.5), dpi=300)
    hourly_wkday = df[df["is_weekend"] == 0].groupby("hour")["traffic_volume"].mean()
    hourly_wknd = df[df["is_weekend"] == 1].groupby("hour")["traffic_volume"].mean()

    plt.plot(hourly_wkday.index, hourly_wkday.values, marker="o", color="#1f77b4", linewidth=2.5, label="Weekday (Mon-Fri)")
    plt.plot(hourly_wknd.index, hourly_wknd.values, marker="s", color="#ff7f0e", linewidth=2.5, linestyle="--", label="Weekend (Sat-Sun)")

    plt.title("Diurnal Traffic Volume Profile: Weekdays vs. Weekends (I-94 Westbound)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour of Day (00:00 - 23:00)", fontsize=11)
    plt.ylabel("Mean Hourly Traffic Volume (veh/hr)", fontsize=11)
    plt.xticks(range(0, 24))
    plt.axvspan(7, 9, color="#1f77b4", alpha=0.12, label="Morning Peak (7-9 AM)")
    plt.axvspan(16, 18, color="#d62728", alpha=0.12, label="Evening Peak (4-6 PM)")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "traffic_by_hour.png")
    plt.close()

    # 2. Traffic by Weekday
    print("Generating plot 2: traffic_by_weekday.png...")
    plt.figure(figsize=(10, 5.5), dpi=300)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Map day_of_week int to weekday name
    day_name_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    df["weekday_name"] = df["day_of_week"].map(day_name_map)

    palette = ["#2b5c8f" if d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] else "#e28743" for d in weekday_order]
    sns.boxplot(
        data=df,
        x="weekday_name",
        y="traffic_volume",
        order=weekday_order,
        palette=palette,
        fliersize=1,
        linewidth=1.2,
    )
    plt.title("Hourly Traffic Volume Distribution Across Days of the Week", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Day of Week", fontsize=11)
    plt.ylabel("Hourly Traffic Volume (veh/hr)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "traffic_by_weekday.png")
    plt.close()

    # 3. Traffic by Weather Condition
    print("Generating plot 3: traffic_by_weather.png...")
    plt.figure(figsize=(11, 5.5), dpi=300)
    top_weather = df["weather_main"].value_counts().nlargest(8).index.tolist()
    weather_df = df[df["weather_main"].isin(top_weather)]
    weather_means = weather_df.groupby("weather_main")["traffic_volume"].mean().sort_values(ascending=False)

    colors = sns.color_palette("crest", n_colors=len(weather_means))
    sns.barplot(
        x=weather_means.index,
        y=weather_means.values,
        palette=colors,
    )
    plt.title("Impact of Prevailing Weather Conditions on Average Traffic Volume", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Dominant Weather Condition", fontsize=11)
    plt.ylabel("Average Traffic Volume (veh/hr)", fontsize=11)
    for i, v in enumerate(weather_means.values):
        plt.text(i, v + 60, f"{int(v):,}", ha="center", fontsize=9.5, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(out_dir / "traffic_by_weather.png")
    plt.close()

    # 4. Traffic by Holiday Status
    print("Generating plot 4: traffic_by_holiday.png...")
    plt.figure(figsize=(9, 5.5), dpi=300)
    df["holiday_category"] = "Regular Day"
    df.loc[df["is_holiday_eve"] == 1, "holiday_category"] = "Holiday Eve"
    df.loc[df["is_holiday"] == 1, "holiday_category"] = "Statutory Holiday"

    holiday_order = ["Regular Day", "Holiday Eve", "Statutory Holiday"]
    palette_hol = {"Regular Day": "#3470a3", "Holiday Eve": "#f39c12", "Statutory Holiday": "#e74c3c"}

    sns.barplot(
        data=df,
        x="holiday_category",
        y="traffic_volume",
        order=holiday_order,
        palette=palette_hol,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
    )
    plt.title("Traffic Volume Shift: Regular vs. Holiday Eve vs. Statutory Holiday", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Calendar Classification", fontsize=11)
    plt.ylabel("Mean Traffic Volume (veh/hr)", fontsize=11)
    cat_means = df.groupby("holiday_category")["traffic_volume"].mean()
    for i, cat in enumerate(holiday_order):
        v = cat_means.get(cat, 0)
        plt.text(i, v + 60, f"{int(v):,} veh/hr", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "traffic_by_holiday.png")
    plt.close()

    # 5. Correlation Heatmap
    print("Generating plot 5: correlation_heatmap.png...")
    plt.figure(figsize=(11, 9), dpi=300)
    corr_features = [
        "traffic_volume",
        "traffic_lag_1h",
        "traffic_lag_2h",
        "traffic_lag_24h",
        "traffic_rolling_mean_3h",
        "traffic_rolling_mean_24h",
        "hour",
        "is_weekend",
        "is_rush_hour",
        "temp_c",
        "weather_severity_index",
        "travel_time_min",
        "accident_risk_index",
    ]
    sub_df = df[corr_features].dropna()
    corr_matrix = sub_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-0.5,
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5},
    )
    plt.title("Multivariate Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()

    print(f"All 5 EDA visualizations successfully saved to {out_dir}/")


if __name__ == "__main__":
    generate_all_plots()

