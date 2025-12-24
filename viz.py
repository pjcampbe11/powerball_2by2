import os
import matplotlib.pyplot as plt

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_heatmap(df_matrix, title, out_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(df_matrix.values, aspect="auto")
    plt.title(title)
    plt.xlabel("Number")
    plt.ylabel("Number")
    plt.xticks(range(len(df_matrix.columns)), df_matrix.columns, rotation=90)
    plt.yticks(range(len(df_matrix.index)), df_matrix.index)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_bar(df, title, out_path, x_col="pair", y_col="value", top_n=10):
    sub = df.head(top_n)
    plt.figure(figsize=(12, 5))
    plt.bar([str(x) for x in sub[x_col]], sub[y_col])
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hist(series, title, out_path, bins=40):
    plt.figure(figsize=(10, 5))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter(x, y, title, out_path, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
