import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def draw_hist(data):
    normal = data.loc[data['Class_att'] == 'Normal']
    abnormal = data.loc[data['Class_att'] == 'Abnormal']

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    ax = axes.ravel()
    for i in range(12):
        _, bins, _ = ax[i].hist(normal.iloc[:, i], bins=20, alpha=0.7)
        ax[i].hist(abnormal.iloc[:, i], bins=bins, alpha=0.7)
        ax[i].set_title(list(normal)[i])

    ax[0].legend(["normal", "abnormal"], loc="best")
    fig.tight_layout()
    plt.show()

def plot_coefficients(feature_names, coef):
    coef = coef.ravel()
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef]
    plt.bar(np.arange(len(coef)), coef, color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(len(coef)), feature_names)
    plt.xlabel("Features")
    plt.ylabel("Coefficients")
    plt.show()

def draw_heatmap(data):
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='viridis')
    plt.show()

def plot_features_importances(feature_names, importances):
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(np.arange(len(importances)), feature_names)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()