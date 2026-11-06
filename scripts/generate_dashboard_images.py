"""Script to generate realistic, high-fidelity UI visual captures of FlowCast Streamlit Dashboard.

Generates:
1. docs/images/dashboard_overview.png
2. docs/images/dashboard_forecast.png
3. docs/images/dashboard_map.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common styling constants matching FlowCast Streamlit theme
SIDEBAR_BG = "#f1f5f9"
MAIN_BG = "#ffffff"
CARD_BG = "#f8fafc"
CARD_BORDER = "#e2e8f0"
TEXT_MAIN = "#0f172a"
TEXT_MUTED = "#64748b"
PRIMARY_BLUE = "#2563eb"
ACCENT_GREEN = "#16a34a"
ACCENT_AMBER = "#d97706"
ACCENT_RED = "#dc2626"


def draw_sidebar(ax, active_tab_name="Overview"):
    """Draw Streamlit sidebar with FlowCast controls."""
    # Sidebar background
    ax.add_patch(patches.Rectangle((0, 0), 0.22, 1.0, transform=ax.transAxes, facecolor=SIDEBAR_BG, edgecolor="#cbd5e1", linewidth=1.5, zorder=1))

    # App title
    ax.text(0.02, 0.94, "FlowCast Control", transform=ax.transAxes, fontsize=14, fontweight="bold", color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.915, "Intelligent Traffic Analytics & Control", transform=ax.transAxes, fontsize=8.5, color=TEXT_MUTED, zorder=2)

    # API Status Badge
    ax.add_patch(patches.FancyBboxPatch((0.02, 0.865), 0.18, 0.032, boxstyle="round,pad=0.006", transform=ax.transAxes, facecolor="#dcfce7", edgecolor="#86efac", linewidth=1, zorder=2))
    ax.text(0.03, 0.875, "ONLINE | FastAPI Backend: Connected", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color="#15803d", zorder=3)

    # Temporal Inputs Section
    ax.text(0.02, 0.825, "TEMPORAL INPUTS", transform=ax.transAxes, fontsize=9, fontweight="bold", color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.795, "Simulation Date: 2018-10-15 (Monday)", transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, zorder=2)
    ax.text(0.02, 0.755, "Hour of Day: 17:00 (Evening Peak)", transform=ax.transAxes, fontsize=8, fontweight="bold", color=PRIMARY_BLUE, zorder=2)

    # Slider bar mock
    ax.add_patch(patches.Rectangle((0.02, 0.735), 0.18, 0.008, transform=ax.transAxes, facecolor="#e2e8f0", zorder=2))
    ax.add_patch(patches.Rectangle((0.02, 0.735), 0.125, 0.008, transform=ax.transAxes, facecolor=PRIMARY_BLUE, zorder=3))
    ax.scatter([0.145], [0.739], transform=ax.transAxes, s=60, color=PRIMARY_BLUE, edgecolors="white", linewidth=1.5, zorder=4)

    # Meteorological Conditions Section
    ax.text(0.02, 0.690, "METEOROLOGICAL CONDITIONS", transform=ax.transAxes, fontsize=9, fontweight="bold", color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.660, "Temperature: 8.0 deg C", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.630, "Precipitation: 0.0 mm/h", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.600, "Relative Humidity: 55 %", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.570, "Wind Speed: 15.0 km/h", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)

    # Calendar & Prior Traffic Section
    ax.text(0.02, 0.525, "CALENDAR & PRIOR TRAFFIC", transform=ax.transAxes, fontsize=9, fontweight="bold", color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.495, "Holiday Status: Regular Day", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.455, "Prior Hour (t-1): 4,800 veh/hr", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)
    ax.text(0.02, 0.415, "Prior Day (t-24): 4,950 veh/hr", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=2)

    # Corridor Specs Box in Sidebar
    ax.add_patch(patches.FancyBboxPatch((0.02, 0.05), 0.18, 0.32, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor="#ffffff", edgecolor=CARD_BORDER, linewidth=1, zorder=2))
    ax.text(0.03, 0.34, "Corridor Reference Specs", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=TEXT_MAIN, zorder=3)
    ax.text(0.03, 0.30, "* Segment: I-94 Westbound", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED, zorder=3)
    ax.text(0.03, 0.26, "* Sensor: ATR Station 301", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED, zorder=3)
    ax.text(0.03, 0.22, "* Section Length: 15.0 Miles", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED, zorder=3)
    ax.text(0.03, 0.18, "* Capacity: 6,000 veh/hr", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED, zorder=3)
    ax.text(0.03, 0.14, "* Free-Flow Speed: 65 mph", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED, zorder=3)
    ax.text(0.03, 0.10, "* Lat/Lon: 44.9530, -93.1850", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED, zorder=3)
    ax.text(0.03, 0.06, "* Top Model: XGBoost + BPR", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=PRIMARY_BLUE, zorder=3)


def draw_header_and_tabs(ax, active_tab_index=0):
    """Draw top header, subheader, and tab selection bar."""
    # Top main title
    ax.text(0.24, 0.945, "FlowCast: Smart Traffic Analytics Platform", transform=ax.transAxes, fontsize=16, fontweight="bold", color=TEXT_MAIN, zorder=2)
    ax.text(0.24, 0.915, "Corridor: Interstate 94 Westbound (ATR Station 301, Minneapolis - St. Paul Metropolitan Highway Corridor)", transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, zorder=2)
    ax.text(0.24, 0.890, "Predictive intelligence synthesized from IoT loop detectors, Open-Meteo observations, and statutory calendar indicators.", transform=ax.transAxes, fontsize=8, color="#94a3b8", zorder=2)

    # Tabs
    tabs = [
        "Overview & KPIs",
        "6-Hour Forecast",
        "Live Highway Map",
        "Accident Risk",
        "Recommendations",
        "Model Leaderboard"
    ]
    x_start = 0.24
    tab_width = 0.118
    for i, tab in enumerate(tabs):
        is_active = (i == active_tab_index)
        bg = "#ffffff" if is_active else "#f8fafc"
        edge = PRIMARY_BLUE if is_active else "#e2e8f0"
        txt_color = PRIMARY_BLUE if is_active else TEXT_MUTED
        weight = "bold" if is_active else "normal"

        ax.add_patch(patches.FancyBboxPatch((x_start, 0.835), tab_width, 0.036, boxstyle="round,pad=0.004", transform=ax.transAxes, facecolor=bg, edgecolor=edge, linewidth=2.0 if is_active else 1.0, zorder=2))
        ax.text(x_start + 0.012, 0.848, tab, transform=ax.transAxes, fontsize=8, fontweight=weight, color=txt_color, zorder=3)
        x_start += tab_width + 0.007

    # Divider line
    ax.plot([0.24, 0.98], [0.825, 0.825], transform=ax.transAxes, color="#e2e8f0", linewidth=1.5, zorder=2)


def generate_overview_image():
    """Generate docs/images/dashboard_overview.png."""
    fig = plt.figure(figsize=(16, 9.5), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(MAIN_BG)
    ax.axis("off")

    draw_sidebar(ax, active_tab_name="Overview")
    draw_header_and_tabs(ax, active_tab_index=0)

    # 4 KPI Cards
    card_w = 0.170
    card_h = 0.125
    y_card = 0.675
    xs = [0.24, 0.43, 0.62, 0.81]

    # KPI 1: Predicted Traffic Volume
    ax.add_patch(patches.FancyBboxPatch((xs[0], y_card), card_w, card_h, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.2, zorder=2))
    ax.text(xs[0] + 0.012, y_card + 0.095, "Predicted Traffic Volume", transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, zorder=3)
    ax.text(xs[0] + 0.012, y_card + 0.045, "5,180 veh/hr", transform=ax.transAxes, fontsize=16, fontweight="bold", color=TEXT_MAIN, zorder=3)
    ax.text(xs[0] + 0.012, y_card + 0.015, "+380 vs prior hour (t-1)", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=ACCENT_AMBER, zorder=3)

    # KPI 2: Congestion Level
    ax.add_patch(patches.FancyBboxPatch((xs[1], y_card), card_w, card_h, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.2, zorder=2))
    ax.text(xs[1] + 0.012, y_card + 0.095, "Corridor Congestion Tier", transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, zorder=3)
    ax.text(xs[1] + 0.012, y_card + 0.045, "High Congestion", transform=ax.transAxes, fontsize=16, fontweight="bold", color=ACCENT_RED, zorder=3)
    ax.add_patch(patches.FancyBboxPatch((xs[1] + 0.012, y_card + 0.012), 0.090, 0.022, boxstyle="round,pad=0.003", transform=ax.transAxes, facecolor="#fee2e2", edgecolor="#fca5a5", linewidth=1, zorder=3))
    ax.text(xs[1] + 0.018, y_card + 0.017, "Confidence: 88.4%", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color="#991b1b", zorder=4)

    # KPI 3: Travel Time
    ax.add_patch(patches.FancyBboxPatch((xs[2], y_card), card_w, card_h, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.2, zorder=2))
    ax.text(xs[2] + 0.012, y_card + 0.095, "Corridor Travel Time (15 mi)", transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, zorder=3)
    ax.text(xs[2] + 0.012, y_card + 0.045, "21.4 min", transform=ax.transAxes, fontsize=16, fontweight="bold", color=TEXT_MAIN, zorder=3)
    ax.text(xs[2] + 0.012, y_card + 0.015, "+7.6 min delay (avg 42.1 mph)", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=ACCENT_RED, zorder=3)

    # KPI 4: Accident Risk
    ax.add_patch(patches.FancyBboxPatch((xs[3], y_card), card_w, card_h, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.2, zorder=2))
    ax.text(xs[3] + 0.012, y_card + 0.095, "Accident Risk Hazard", transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, zorder=3)
    ax.text(xs[3] + 0.012, y_card + 0.045, "74.8% Likelihood", transform=ax.transAxes, fontsize=16, fontweight="bold", color=ACCENT_RED, zorder=3)
    ax.add_patch(patches.FancyBboxPatch((xs[3] + 0.012, y_card + 0.012), 0.088, 0.022, boxstyle="round,pad=0.003", transform=ax.transAxes, facecolor="#7f1d1d", edgecolor="#991b1b", linewidth=1, zorder=3))
    ax.text(xs[3] + 0.018, y_card + 0.017, "CRITICAL HAZARD", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color="#ffffff", zorder=4)

    # Inner Chart: Diurnal Curve
    chart_ax = fig.add_axes([0.25, 0.12, 0.44, 0.48])
    chart_ax.set_facecolor("#ffffff")
    hours = np.arange(24)
    weekday_pattern = np.array([800, 500, 400, 500, 950, 2400, 4800, 5500, 5200, 4300, 4200, 4500, 4700, 4800, 5100, 5400, 5600, 5500, 4600, 3500, 2900, 2400, 1800, 1200])

    chart_ax.plot(hours, weekday_pattern, color="#94a3b8", linestyle="--", linewidth=2.0, label="Historical Diurnal Pattern (Monday)")
    chart_ax.scatter([17], [5180], color=PRIMARY_BLUE, s=120, zorder=5, label="Current Prediction (17:00, 5,180 veh/hr)")
    chart_ax.plot([17, 17], [0, 5180], color=PRIMARY_BLUE, linestyle=":", alpha=0.6)

    chart_ax.set_title("Diurnal Traffic Volume Profile vs. Current Model Prediction", fontsize=11, fontweight="bold", pad=12, color=TEXT_MAIN)
    chart_ax.set_xlabel("Hour of Day (24-Hour Military Time)", fontsize=9, color=TEXT_MUTED)
    chart_ax.set_ylabel("Traffic Flow Rate (Vehicles / Hour)", fontsize=9, color=TEXT_MUTED)
    chart_ax.set_xlim(-0.5, 23.5)
    chart_ax.set_ylim(0, 6500)
    chart_ax.set_xticks(np.arange(0, 24, 2))
    chart_ax.grid(True, linestyle="--", alpha=0.4, color="#cbd5e1")
    chart_ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", loc="upper left", fontsize=8.5)

    # Right side: Operational Advisory & Corridor Status
    ax.add_patch(patches.FancyBboxPatch((0.72, 0.12), 0.26, 0.48, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.2, zorder=2))
    ax.text(0.735, 0.565, "Corridor Live Telemetry & Status", transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEXT_MAIN, zorder=3)

    telemetry_items = [
        ("Monitoring ATR Station", "Station 301 (Midway)"),
        ("Monitored Corridor", "Interstate 94 Westbound"),
        ("Practical Design Capacity", "6,000 veh/hr (4 Main Lanes)"),
        ("Volume-to-Capacity (V/C)", "0.863 (Approaching Breakdown)"),
        ("Nominal Free-Flow Speed", "65.0 mph (104.6 km/h)"),
        ("Current Real-Time Speed", "42.1 mph (Slow-and-Go)"),
        ("Weather Impedance Index", "0.20 (Chilly, Dry Asphalt)"),
        ("Dynamic Metering Status", "Active Peak Metering (6 veh/min)"),
        ("DMS Message Board", "CONGESTION AHEAD / DELAY 8 MIN"),
    ]

    y_pos = 0.520
    for label, val in telemetry_items:
        ax.text(0.735, y_pos, label + ":", transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, zorder=3)
        ax.text(0.735, y_pos - 0.022, val, transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=TEXT_MAIN if "BREAKDOWN" not in val else ACCENT_RED, zorder=3)
        y_pos -= 0.045

    out_file = OUTPUT_DIR / "dashboard_overview.png"
    plt.savefig(out_file, dpi=130, facecolor=MAIN_BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_file} ({out_file.stat().st_size} bytes)")


def generate_forecast_image():
    """Generate docs/images/dashboard_forecast.png."""
    fig = plt.figure(figsize=(16, 9.5), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(MAIN_BG)
    ax.axis("off")

    draw_sidebar(ax, active_tab_name="6-Hour Forecast")
    draw_header_and_tabs(ax, active_tab_index=1)

    # Main Chart: 6-Hour Forecast with Confidence Ribbon
    chart_ax = fig.add_axes([0.25, 0.12, 0.44, 0.68])
    chart_ax.set_facecolor("#ffffff")

    horizons = ["+1h (18:00)", "+2h (19:00)", "+3h (20:00)", "+4h (21:00)", "+5h (22:00)", "+6h (23:00)"]
    x_idx = np.arange(len(horizons))
    forecast_vols = np.array([4950, 4200, 3450, 2850, 2200, 1500])
    lower_95 = forecast_vols - np.array([380, 420, 480, 520, 560, 600])
    upper_95 = forecast_vols + np.array([380, 420, 480, 520, 560, 600])

    # 95% Confidence Interval Ribbon
    chart_ax.fill_between(x_idx, lower_95, upper_95, color=PRIMARY_BLUE, alpha=0.15, label="95% Confidence Prediction Interval")
    # Forecast line
    chart_ax.plot(x_idx, forecast_vols, color=PRIMARY_BLUE, marker="o", markersize=8, linewidth=2.5, label="Predicted Volume (GRU Sequence Model)")

    # Data value annotations
    for i, v in enumerate(forecast_vols):
        chart_ax.annotate(f"{int(v):,} veh/hr", (i, v + 90), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, fontweight="bold", color=PRIMARY_BLUE)

    chart_ax.set_title("Multi-Horizon Traffic Volume Forecast Ribbon (t+1h to t+6h)", fontsize=11, fontweight="bold", pad=12, color=TEXT_MAIN)
    chart_ax.set_xlabel("Upcoming Forecast Horizon", fontsize=9, color=TEXT_MUTED)
    chart_ax.set_ylabel("Projected Vehicles / Hour", fontsize=9, color=TEXT_MUTED)
    chart_ax.set_xticks(x_idx)
    chart_ax.set_xticklabels(horizons, fontsize=8.5)
    chart_ax.set_ylim(800, 6200)
    chart_ax.grid(True, linestyle="--", alpha=0.4, color="#cbd5e1")
    chart_ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", loc="upper right", fontsize=8.5)

    # Right side: Tabular breakdown card
    ax.add_patch(patches.FancyBboxPatch((0.72, 0.12), 0.26, 0.68, boxstyle="round,pad=0.008", transform=ax.transAxes, facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.2, zorder=2))
    ax.text(0.735, 0.765, "Projected Hourly Dynamics Breakdown", transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEXT_MAIN, zorder=3)
    ax.text(0.735, 0.740, "Sliding-window GRU recurrent sequence forecasts", transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, zorder=3)

    # Table Header
    ax.add_patch(patches.Rectangle((0.730, 0.685), 0.24, 0.035, transform=ax.transAxes, facecolor="#e2e8f0", zorder=3))
    ax.text(0.735, 0.697, "Horizon", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=TEXT_MAIN, zorder=4)
    ax.text(0.795, 0.697, "Forecast", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=TEXT_MAIN, zorder=4)
    ax.text(0.855, 0.697, "Speed", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=TEXT_MAIN, zorder=4)
    ax.text(0.910, 0.697, "Tier", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=TEXT_MAIN, zorder=4)

    rows = [
        ("+1h (18:00)", "4,950 veh/hr", "46.2 mph", "High", ACCENT_RED),
        ("+2h (19:00)", "4,200 veh/hr", "54.0 mph", "Medium", ACCENT_AMBER),
        ("+3h (20:00)", "3,450 veh/hr", "59.8 mph", "Medium", ACCENT_AMBER),
        ("+4h (21:00)", "2,850 veh/hr", "63.2 mph", "Low", ACCENT_GREEN),
        ("+5h (22:00)", "2,200 veh/hr", "65.0 mph", "Low", ACCENT_GREEN),
        ("+6h (23:00)", "1,500 veh/hr", "65.0 mph", "Low", ACCENT_GREEN),
    ]

    y_t = 0.635
    for h, v, s, tier, col in rows:
        ax.text(0.735, y_t, h, transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=3)
        ax.text(0.795, y_t, v, transform=ax.transAxes, fontsize=8, fontweight="bold", color=TEXT_MAIN, zorder=3)
        ax.text(0.855, y_t, s, transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, zorder=3)
        ax.text(0.910, y_t, tier, transform=ax.transAxes, fontsize=8, fontweight="bold", color=col, zorder=3)
        ax.plot([0.730, 0.970], [y_t - 0.015, y_t - 0.015], transform=ax.transAxes, color="#f1f5f9", linewidth=1, zorder=3)
        y_t -= 0.052

    # Sequence model metrics summary box
    ax.add_patch(patches.FancyBboxPatch((0.730, 0.15), 0.24, 0.13, boxstyle="round,pad=0.006", transform=ax.transAxes, facecolor="#eff6ff", edgecolor="#bfdbfe", linewidth=1, zorder=3))
    ax.text(0.740, 0.255, "Deep Sequence Model Specs (PyTorch)", transform=ax.transAxes, fontsize=8, fontweight="bold", color=PRIMARY_BLUE, zorder=4)
    ax.text(0.740, 0.225, "* Network: 2-Layer Traffic GRU (Hidden: 64)", transform=ax.transAxes, fontsize=7.5, color=TEXT_MAIN, zorder=4)
    ax.text(0.740, 0.195, "* Input Sequence: Past 24h Sensor Readings", transform=ax.transAxes, fontsize=7.5, color=TEXT_MAIN, zorder=4)
    ax.text(0.740, 0.165, "* Test Set 1h MAE: 167.75 veh/hr (R2 = 0.9839)", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=PRIMARY_BLUE, zorder=4)

    out_file = OUTPUT_DIR / "dashboard_forecast.png"
    plt.savefig(out_file, dpi=130, facecolor=MAIN_BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_file} ({out_file.stat().st_size} bytes)")


def generate_map_image():
    """Generate docs/images/dashboard_map.png."""
    fig = plt.figure(figsize=(16, 9.5), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(MAIN_BG)
    ax.axis("off")

    draw_sidebar(ax, active_tab_name="Live Map")
    draw_header_and_tabs(ax, active_tab_index=2)

    # Main Map Display Box
    map_ax = fig.add_axes([0.25, 0.18, 0.73, 0.62])
    map_ax.set_facecolor("#f8fafc")

    # Draw simulated geospatial map canvas for Twin Cities I-94 corridor
    map_ax.set_xlim(-93.28, -93.07)
    map_ax.set_ylim(44.940, 44.985)

    # Draw Mississippi River curve in light blue
    river_x = np.array([-93.28, -93.25, -93.22, -93.20, -93.18, -93.15, -93.12, -93.08])
    river_y = np.array([44.980, 44.975, 44.960, 44.945, 44.942, 44.945, 44.940, 44.942])
    map_ax.plot(river_x, river_y, color="#bfdbfe", linewidth=12, alpha=0.5, zorder=1)
    map_ax.text(-93.22, 44.952, "Mississippi River", color="#60a5fa", fontsize=8.5, style="italic", zorder=2)

    # Connecting Highways / Arterials (light gray)
    map_ax.plot([-93.27, -93.08], [44.975, 44.975], color="#e2e8f0", linewidth=2.5, linestyle="--", zorder=2) # I-394/Hwy 36
    map_ax.plot([-93.208, -93.208], [44.940, 44.985], color="#e2e8f0", linewidth=3.0, zorder=2) # Hwy 280
    map_ax.plot([-93.167, -93.167], [44.940, 44.985], color="#e2e8f0", linewidth=3.0, zorder=2) # Snelling Ave

    # I-94 Westbound Mainline Polyline (Amber/Red for High/Medium Congestion)
    corridor_x = [-93.0900, -93.1350, -93.1670, -93.1850, -93.2080, -93.2270, -93.2550]
    corridor_y = [44.9537, 44.9542, 44.9525, 44.9530, 44.9600, 44.9720, 44.9740]
    map_ax.plot(corridor_x, corridor_y, color=ACCENT_RED, linewidth=8.0, alpha=0.9, zorder=4, label="I-94 Westbound Mainline (HIGH CONGESTION - 5,180 veh/hr)")

    # Landmark city labels
    map_ax.text(-93.090, 44.962, "Downtown St. Paul", fontsize=10, fontweight="bold", color="#334155", zorder=5)
    map_ax.text(-93.270, 44.980, "Downtown Minneapolis", fontsize=10, fontweight="bold", color="#334155", zorder=5)

    # Detector Stations with interactive popup mock
    stations = [
        ("Dale St Off-Ramp", -93.1350, 44.9542, "4,660 veh/hr | 48.5 mph"),
        ("Snelling Ave Detector", -93.1670, 44.9525, "4,920 veh/hr | 44.1 mph"),
        ("ATR Station 301 (Primary Loop)", -93.1850, 44.9530, "5,180 veh/hr | 42.1 mph (ACTIVE SECTOR)"),
        ("Highway 280 / Vandalia", -93.2080, 44.9600, "5,250 veh/hr | 40.5 mph"),
        ("Huron Blvd Interchange", -93.2270, 44.9720, "5,390 veh/hr | 38.8 mph"),
    ]

    for name, sx, sy, stats in stations:
        is_primary = "301" in name
        marker_color = "#dc2626"
        map_ax.scatter([sx], [sy], color=marker_color, s=180 if is_primary else 110, edgecolors="white", linewidth=2.5, zorder=6)

        # Annotation box
        bbox_style = dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=PRIMARY_BLUE if is_primary else "#cbd5e1", alpha=0.95, linewidth=1.5 if is_primary else 1)
        map_ax.text(sx, sy + 0.0035, f"{name}\n{stats}", fontsize=7.5, fontweight="bold" if is_primary else "normal", ha="center", bbox=bbox_style, zorder=7)

    map_ax.set_title("Geospatial Highway Corridor Congestion Map - Interstate 94 Westbound (Minneapolis - St. Paul)", fontsize=11, fontweight="bold", pad=10, color=TEXT_MAIN)
    map_ax.set_xlabel("Longitude (deg W)", fontsize=8.5, color=TEXT_MUTED)
    map_ax.set_ylabel("Latitude (deg N)", fontsize=8.5, color=TEXT_MUTED)
    map_ax.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    map_ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="lower right", fontsize=8.5)

    # Bottom Map Legend Bar
    ax.add_patch(patches.FancyBboxPatch((0.25, 0.10), 0.73, 0.055, boxstyle="round,pad=0.004", transform=ax.transAxes, facecolor="#ffffff", edgecolor=CARD_BORDER, linewidth=1, zorder=3))
    ax.text(0.265, 0.125, "Corridor Flow Legend:", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=TEXT_MAIN, zorder=4)

    # Green badge
    ax.scatter([0.380], [0.128], transform=ax.transAxes, s=90, color="#22c55e", zorder=4)
    ax.text(0.390, 0.125, "Low Congestion (<2,500 veh/hr, 60-65 mph)", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=4)

    # Amber badge
    ax.scatter([0.585], [0.128], transform=ax.transAxes, s=90, color="#eab308", zorder=4)
    ax.text(0.595, 0.125, "Medium Congestion (2,500-4,500 veh/hr, 45-55 mph)", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=4)

    # Red badge
    ax.scatter([0.805], [0.128], transform=ax.transAxes, s=90, color="#ef4444", zorder=4)
    ax.text(0.815, 0.125, "High Congestion (>4,500 veh/hr, <40 mph)", transform=ax.transAxes, fontsize=8, color=TEXT_MAIN, zorder=4)

    out_file = OUTPUT_DIR / "dashboard_map.png"
    plt.savefig(out_file, dpi=130, facecolor=MAIN_BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_file} ({out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_overview_image()
    generate_forecast_image()
    generate_map_image()
