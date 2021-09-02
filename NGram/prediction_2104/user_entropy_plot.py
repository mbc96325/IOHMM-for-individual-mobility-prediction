import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


colors = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099"]
labels = [r'$o$', r'$d$', r'$t$', r'$\Delta t$']
df = pd.read_csv('../output/user_entropy.csv')


def joint_density_plot(df):
	plt.figure(figsize=(18,6))
	plt.subplot(131)
	sns.kdeplot(df['o_ent'], df['d_ent'], shade=True, cmap='Reds')
	plt.subplot(132)
	sns.kdeplot(df['o_ent'], df['t_ent'], shade=True, cmap='Reds')
	plt.subplot(133)
	sns.kdeplot(df['d_ent'], df['t_ent'], shade=True, cmap='Reds')
	plt.show()


def density_plot(df):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid")
	fig, ax = plt.subplots(figsize=(16, 8))
	sns.kdeplot(df['o_ent'], ax=ax, shade=True, color=colors[0], label=labels[0])
	sns.kdeplot(df['d_ent'], ax=ax, shade=True, color=colors[1], label=labels[1])
	sns.kdeplot(df['t_ent'], ax=ax, shade=True, color=colors[2], label=labels[2])
	sns.kdeplot(df['dt_ent'], ax=ax, shade=True, color=colors[3], label=labels[3])
	plt.axvline(df['o_ent'].mean(), color=colors[0], linestyle='dashed', linewidth=2)
	plt.axvline(df['d_ent'].mean(), color=colors[1], linestyle='dashed', linewidth=2)	
	plt.axvline(df['t_ent'].mean(), color=colors[2], linestyle='dashed', linewidth=2)
	plt.axvline(df['dt_ent'].mean(), color=colors[3], linestyle='dashed', linewidth=2)
	plt.xlim(0, 8)
	plt.xlabel('Entropy', fontsize=16)
	plt.ylabel('Probability', fontsize=16)
	ax.legend(fontsize=18)
	plt.savefig('../img/entropy_dbn.png', dpi=300)


density_plot(df)
