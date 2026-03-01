import pandas as pd
import matplotlib.pyplot as plt

csv_path = ''
out_path = ''

full_df = pd.read_csv(csv_path)

# 1 value per speaker and training size (avg over layers/sensors)
per_speaker = (
    full_df
    .groupby(['speaker', 'training_size_seconds'])['pearson_correlation']
    .mean()
    .reset_index()
)

# aggregation across speakers per training size
summary = (
    per_speaker
    .groupby('training_size_seconds')['pearson_correlation']
    .agg(mean='mean', min='min', max='max', std='std', n='count')
    .reset_index()
    .sort_values('training_size_seconds')
)

# even spacing on x-axis
ticks = sorted(summary['training_size_seconds'].unique())
x_positions = list(range(len(ticks)))
summary = summary.set_index('training_size_seconds').loc[ticks].reset_index()

mean = summary['mean'].to_numpy()
ymin = summary['min'].to_numpy()
ymax = summary['max'].to_numpy()
std = summary['std'].fillna(0.0).to_numpy()

mean_color = "#0072B2"
shade_color = "#56B4E9"

plt.figure(figsize=(10, 6))

# ADD: individual speaker curves (ultra thin, background)

# map training size to x position
x_map = {t: i for i, t in enumerate(ticks)}

for speaker_id, data in per_speaker.groupby('speaker'):
    data = data.sort_values('training_size_seconds')
    
    x_vals = [x_map[t] for t in data['training_size_seconds']]
    y_vals = data['pearson_correlation'].to_numpy()
    
    # linestyle by language
    if speaker_id.startswith('fin_'):
        linestyle = '-'
    elif speaker_id.startswith('rus_'):
        linestyle = ':'
    else:
        linestyle = '-'
    
    plt.plot(
        x_vals,
        y_vals,
        color='gray',
        linewidth=0.9,
        alpha=0.35,
        linestyle=linestyle
    )

plt.fill_between(
    x_positions, ymin, ymax,
    color=shade_color,
    alpha=0.18,
    label='Min–Max across speakers'
)

plt.fill_between(
    x_positions,
    mean - std,
    mean + std,
    color=mean_color,
    alpha=0.20,
    label='±1 std'
)

plt.plot(
    x_positions, mean,
    marker='o',
    linewidth=2.5,
    color=mean_color,
    label='Mean across speakers'
)

if 300 in ticks:
    idx_300 = ticks.index(300)
    plt.axvline(
        x=idx_300,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.6,
        label='Plateau at 300s'
    )

plt.xlabel('Training Size (s)')
plt.ylabel('Pearson Correlation')
plt.title('Training Size vs Pearson Correlation')
plt.grid(True, alpha=0.3)

ax = plt.gca()
ax.set_xticks(x_positions)
ax.set_xticklabels([str(int(t)) for t in ticks], rotation=0)

handles, labels = ax.get_legend_handles_labels()
seen = set()
unique_handles, unique_labels = [], []
for h, lab in zip(handles, labels):
    if lab not in seen:
        unique_handles.append(h)
        unique_labels.append(lab)
        seen.add(lab)
ax.legend(unique_handles, unique_labels)

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()

print("\nSummary per training size:")
print(summary.to_string(index=False))
