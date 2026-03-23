import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ==========================================
# 1. 資料輸入 (依據您的表格數據)
# ==========================================
# 類別名稱 (按照您論文中提到的順序或數量排列)
categories = [
    "Healthy", "Eczema", "Psoriasis", 
     "SL", "Nevus", "SK", "Vitiligo"
]

# 各類別對應張數
data_counts = [1011, 355, 322,  227,  204, 172, 150]

OUTPUT_FILE = "oct_class_distribution.png"

# ==========================================
# 2. 配色與樣式 (維持您的專業深藍)
# ==========================================
color_main = '#4C72B0'  # 專業深藍 (Tuned style)

# ==========================================
# 3. 繪圖
# ==========================================
def plot_oct_distribution():
    plt.style.use("default")
    
    # 調整 rcParams 以符合您的論文風格
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })

    # 設定畫布大小 (維持與您的 Heatmap 比例一致)
    fig, ax = plt.subplots(figsize=(10, 5.5)) 

    x = np.arange(len(categories))
    width = 0.6  # 單一長條圖，寬度可以稍微加寬一點點

    # 繪製柱狀圖
    rects = ax.bar(
        x, data_counts, width,
        color=color_main, edgecolor="black", linewidth=1.5, zorder=3
    )

    # 軸標籤設定
    ax.set_ylabel("Number of Images", fontsize=20, labelpad=12)
    
    # X 軸設定 (分類標籤)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=15, rotation=0, ha='center')
    
    # 刻度大小設定
    ax.tick_params(axis="x", labelsize=14, pad=8) 
    ax.tick_params(axis="y", labelsize=16)

    # Y 軸刻度與範圍 (自動預留頂部空間標註數字)
    max_height = max(data_counts)
    ax.set_ylim(0, max_height * 1.15) 
    ax.yaxis.set_major_locator(MultipleLocator(200)) # 根據數據級距調整為 200

    # 細節美化 (去框、格線)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 數字標註 (Auto Label)
    def autolabel(rects):
        for r in rects:
            h = r.get_height()
            ax.annotate(
                f"{int(h)}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 5), 
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=13, fontweight='bold', color="black"
            )

    autolabel(rects)

    plt.tight_layout()
    
    # 同時儲存 PNG 與 PDF
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor='white')
    pdf_output = OUTPUT_FILE.replace('.png', '.pdf')
    plt.savefig(pdf_output, bbox_inches="tight", facecolor='white')

    plt.close()
    print(f"Plot saved to {OUTPUT_FILE} and PDF: {pdf_output}")

if __name__ == "__main__":
    plot_oct_distribution()