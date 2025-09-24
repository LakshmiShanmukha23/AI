# File: plot_results.py
# Simple plotting script to visualize a CSV of results using matplotlib.
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path='sample_results.csv'):
    df = pd.read_csv(csv_path)
    # accuracy bar chart
    plt.figure(figsize=(8,4))
    plt.bar(df['experiment'], df['accuracy'])
    plt.title('Accuracy by Experiment')
    plt.ylim(0,1)
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('accuracy_by_experiment.png')
    print('Saved accuracy_by_experiment.png')

    # latency vs accuracy scatter
    plt.figure(figsize=(6,4))
    plt.scatter(df['latency_s'], df['accuracy'])
    for i,row in df.iterrows():
        plt.annotate(row['experiment'], (row['latency_s'], row['accuracy']))
    plt.xlabel('Latency (s)')
    plt.ylabel('Accuracy')
    plt.title('Latency vs Accuracy')
    plt.tight_layout()
    plt.savefig('latency_vs_accuracy.png')
    print('Saved latency_vs_accuracy.png')

if __name__ == '__main__':
    plot_metrics()
