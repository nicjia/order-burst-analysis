import matplotlib.pyplot as plt
import seaborn as sns

# Academic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# The fresh data from your dynamic rolling-window results
horizons = ['1 Min', '3 Min', '5 Min', '10 Min', 'Market Close']
accuracy = [68.1, 61.2, 59.1, 54.8, 52.8]

plt.figure(figsize=(8, 5))

# Plot the decay line
plt.plot(horizons, accuracy, marker='o', markersize=8, linewidth=2.5, color='#2c3e50', label='Random Forest Directional Accuracy')

# Baseline (Random Guessing = 50%)
plt.axhline(y=50.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Random Baseline (50%)')

# Formatting the chart
plt.title('Out-of-Sample Temporal Decay of Informational Shockwaves', pad=15, fontweight='bold')
plt.ylabel('Directional Accuracy (%)')
plt.xlabel('Prediction Horizon')
plt.ylim(45, 75)

# Annotate the specific values on the dots
for i, txt in enumerate(accuracy):
    plt.annotate(f"{txt}%", (horizons[i], accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend()
plt.tight_layout()

# Save for Overleaf
plt.savefig('temporal_decay.png', dpi=300, bbox_inches='tight')
print("New graph saved successfully as temporal_decay.png")