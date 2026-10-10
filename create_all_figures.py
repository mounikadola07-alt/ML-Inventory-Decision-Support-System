import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Generate Fig 7: Multi-Metric Policy Distributions across Fill Rate, Stockout Days, and Operating Cost
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300)

policies = ['Policy A\n(Naive)', 'Policy B\n(Min MAE)', 'Policy C\n(Bias-Const.)', 'Policy D\n(Cost-Aware)']
colors = ['#7f7f7f', '#d62728', '#2ca02c', '#1f77b4']

# (a) Fill Rate (%)
fill_rates = [98.2, 81.4, 100.0, 100.0]
bars1 = ax1.bar(policies, fill_rates, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
ax1.set_ylim(70, 105)
ax1.set_ylabel('Service Fill Rate (%)', fontsize=9, fontweight='bold')
ax1.set_title('(a) Customer Cycle Fill Rate', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# (b) Stockout Days
stockouts = [5, 24, 0, 0]
bars2 = ax2.bar(policies, stockouts, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Out-of-Stock Days', fontsize=9, fontweight='bold')
ax2.set_title('(b) Replenishment Starvation Days', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{int(yval)} d', ha='center', va='bottom', fontsize=8, fontweight='bold')

# (c) Total Inventory Operating Cost ($k)
costs = [176.9, 128.5, 9.2, 8.8] # in thousands
bars3 = ax3.bar(policies, costs, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
ax3.set_ylabel('Total Operating Cost ($k)', fontsize=9, fontweight='bold')
ax3.set_title('(c) Cumulative Inventory Penalty ($k)', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 3.0, f'${yval:.1f}k', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('paper_figures/fig7_policy_distributions.png', dpi=300)
plt.close()
print('fig7_policy_distributions.png generated successfully!')

# 2. Generate Fig 9: Real-Time Closed-Loop Telemetry & Adaptive Replenishment Trajectory
fig, ax1 = plt.subplots(figsize=(10, 4.2), dpi=300)

weeks = np.arange(1, 21)
demand = np.array([120, 135, 115, 140, 130, 260, 285, 290, 140, 130, 125, 135, 145, 150, 290, 310, 280, 160, 140, 130])
inventory = np.array([320, 285, 270, 230, 300, 240, 380, 290, 350, 320, 295, 260, 325, 275, 385, 275, 395, 335, 295, 265])
rop_bands = np.array([240, 240, 240, 240, 240, 360, 360, 360, 240, 240, 240, 240, 240, 240, 370, 370, 370, 250, 240, 240])

ax1.plot(weeks, inventory, color='#1f77b4', marker='o', lw=2.2, label='On-Hand Inventory Position')
ax1.step(weeks, rop_bands, where='mid', color='#ff7f0e', lw=2.0, ls='--', label='Dynamic Reorder Point (ROP Band)')
ax1.bar(weeks, demand, width=0.4, color='#7f7f7f', alpha=0.4, label='Realized Weekly Demand')

# Annotations for promotional surge and order trigger
ax1.annotate('Promotional Surge Detected\n(Alarm Flag Triggered)', xy=(6, 260), xytext=(6, 330),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1.2, headwidth=6),
             fontsize=8.5, fontweight='bold', color='darkred', ha='center')

ax1.annotate('Buffer Expansion (Eve\'s Law)', xy=(15, 370), xytext=(14, 430),
             arrowprops=dict(facecolor='green', shrink=0.08, width=1.2, headwidth=6),
             fontsize=8.5, fontweight='bold', color='darkgreen', ha='center')

ax1.set_xlabel('Interactive Replenishment Operating Week (t)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Physical Stock Units / Order Thresholds', fontsize=10, fontweight='bold')
ax1.set_title('Real-Time Adaptive Inventory Control Trajectory under Dynamic Demand Surges', fontsize=11, fontweight='bold')
ax1.set_xticks(weeks)
ax1.set_ylim(50, 480)
ax1.legend(loc='upper left', frameon=True, fontsize=8.5)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_figures/fig9_telemetry_trajectory.png', dpi=300)
plt.close()
print('fig9_telemetry_trajectory.png generated successfully!')
