import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_accuracy_vs_resolution(df, save_path="results/plots/accuracy_vs_resolution.png"):
    plt.figure(figsize=(5, 3))
    plt.plot(df['Resolution'], df['Accuracy'], marker='o', color='green')
    plt.title("Accuracy vs Input Resolution")
    plt.xlabel("Input Resolution")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_time_vs_resolution(df, save_path="results/plots/training_time_vs_resolution.png"):
    df['TrainingTimeMin'] = df['TrainingTime'] / 60
    plt.figure(figsize=(5, 3))
    bars = plt.bar(df['Resolution'], df['TrainingTimeMin'], color='#4682B4')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                 f"{height:.1f} min", ha='center', va='bottom', fontsize=8)

    plt.title("Czas treningu vs rozdzielczość")
    plt.xlabel("Rozdzielczość (px)")
    plt.ylabel("Czas treningu (min)")
    plt.ylim(0, df['TrainingTimeMin'].max() + 1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    os.makedirs("results/plots", exist_ok=True)
    df = pd.read_csv('results/logs.csv')
    df = df.sort_values('Resolution')

    plot_accuracy_vs_resolution(df)
    plot_training_time_vs_resolution(df)

if __name__ == "__main__":
    main()
