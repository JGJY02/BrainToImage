import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import os

main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

results_dir = "results/Transformer_512_Large_dual"
type_of_class = "secondary"

data = pd.read_csv(f"{results_dir}/{type_of_class}_IG_res_summarize_channel.csv")
# annotation_bar = [row[-1] for row in data]
print(data.shape)


plt.figure(figsize=(15, 6))
sb.heatmap(data.drop(columns=['number', 'Unnamed: 0']), annot=False, linewidths=.5,
           cmap="YlGnBu", cbar_kws={'label': 'Average Attribution'})

plt.savefig(f'{results_dir}/{type_of_class}_heatmap.png', format='png', bbox_inches='tight')
plt.close()

data = pd.read_csv(f"{results_dir}/primary_IG_res_per_channel.csv")
grouped = data.drop(columns='number').groupby(['Unnamed: 0']).mean()
max_values = grouped.max()
plt.figure(figsize=(15, 6))
sb.heatmap(grouped, annot=False,linewidths=.5,
           cmap="YlGnBu", cbar_kws={'label': 'Average Attribution'})
plt.savefig(f'{results_dir}/{type_of_class}_theMostImportantTime.png', format='png', bbox_inches='tight')
plt.close()

sb.set_style("whitegrid")
data = pd.read_csv(f"{results_dir}/{type_of_class}_IG_res_per_channel.csv")
grouped = data.drop(columns=['number']).groupby(['Unnamed: 0']).mean()
# columns_to_color = ['EEG.Fp1', 'EEG.FC4', 'EEG.FT7', 'EEG.F5','EEG.AF4','EEG.F3', 'EEG.TP9', 'EEG.P7', 'EEG.O1', 'EEG.Oz', 'EEG.O2','EEG.TP10', 'EEG.FCz']
# columns_to_color = ['EEG.Fp1', 'EEG.F3', 'EEG.TP9', 'EEG.TP10', 'EEG.T8', 'EEG.F5'] # For primary
columns_to_color = ['EEG.Fp1', 'EEG.F3', 'EEG.TP9', 'EEG.TP10', 'EEG.T8', 'EEG.F5'] # For Secondary

for column in grouped.columns:
    if column in columns_to_color:
        plt.plot(grouped.index, grouped[column], marker='o', label=column)
    else:
        plt.plot(grouped.index, grouped[column], marker='o', linestyle='--', color='gray', alpha=0.2)

plt.title('Mean Values per Time Step')
plt.xlabel('Time Step')
plt.ylabel('Attribution Mean Value')
plt.grid(True)
plt.legend()
plt.savefig(f'{results_dir}/{type_of_class}_TimelineImportance.png', format='png', bbox_inches='tight')