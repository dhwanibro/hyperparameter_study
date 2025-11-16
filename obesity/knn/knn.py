import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ### The target variable is "NObeyesdad" (obesity level)
# Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

def load_and_preprocess(filepath, target_col):
    """Load data and do preprocessing"""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()  # drop rows with missing values
        print(f"Shape after dropping NaN: {df.shape}")
    else:
        print("No missing values found")
    
    # check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
    
    # encode categorical variables
    # columns like Gender, SMOKE, CAEC, CALC, MTRANS are categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)  # don't encode target yet
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col}")
    
    # separate features and target
    X = df.drop(target_col, axis=1)
    Y = df[target_col]
    
    # encode target variable
    Y = le.fit_transform(Y)
    
    # check class balance
    print(f"\nClass distribution:")
    unique, counts = np.unique(Y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y)*100:.2f}%)")
    
    return X, Y

def split_and_scale(X, Y):
    """Split data and scale features"""
    # if OG data has imbalance then test and train data has same imbalance ratio using stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # creating an instance of the standardscaler class
    # scales features to mean=0, std=1 so extreme values don't distort training
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    # the scaler object stores the mean and std of every column inside itself. transform just applies it
    
    # applying the train data's mean and sd to this, will not fit it cause that causes leakage
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test, k_values=[1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 50]):
    """
    Test sensitivity to n_neighbors (k) hyperparameter
    n_neighbors (k) = number of nearest neighbors to consider for voting
    - k=1 → uses only closest neighbor → very sensitive to noise → overfits
    - k=3-7 → often works well, good balance
    - k large (30+) → smooths decision boundary → may underfit
    
    Rule of thumb: k should be odd for binary classification to avoid ties
    k should be sqrt(n_samples) as starting point
    """
    test_acc_list = []
    k_results = {}
    
    for k in k_values:
        # KNeighborsClassifier finds k nearest neighbors and does majority vote
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        test_acc_list.append(test_acc)
        k_results[k] = {'train': train_acc, 'test': test_acc}
        
        print(f"k={k}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    min_acc = min(test_acc_list)
    max_acc = max(test_acc_list)
    
    relative_change = (max_acc - min_acc) / min_acc
    percent_change = relative_change * 100
    
    print(f"\nn_neighbors Sensitivity: {percent_change:.4f}%")
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, test_acc_list, marker="o", label="Test Accuracy")
    plt.xlabel("n_neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity to n_neighbors")
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_k_sensitivity_plot.png", dpi=150, bbox_inches='tight')
  #  plt.show()
    
    return k_results, min_acc, max_acc, percent_change

def test_weights_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    """
    Test sensitivity to weights hyperparameter
    weights = how to weight the neighbors when voting
    - 'uniform': All neighbors vote equally (default)
    - 'distance': Closer neighbors have more influence
      → weight = 1 / distance
      → points that are closer matter more
    
    'distance' can help when you have varying density of data
    """
    weights_options = ['uniform', 'distance']
    results = {}
    
    for weight in weights_options:
        model = KNeighborsClassifier(n_neighbors=k, weights=weight)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        results[weight] = {'train': train_acc, 'test': test_acc}
        print(f"{weight}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # calculate difference
    test_accs = [results[w]['test'] for w in weights_options]
    acc_diff = abs(test_accs[0] - test_accs[1])
    relative_diff = (acc_diff / min(test_accs)) * 100
    
    # bar plot
    plt.figure(figsize=(8, 5))
    test_accs_list = [results[w]['test'] for w in weights_options]
    
    plt.bar(weights_options, test_accs_list, color=['lightcoral', 'skyblue'])
    plt.ylabel("Test Accuracy")
    plt.title(f"Weights Comparison (k={k})")
    plt.ylim([min(test_accs_list) - 0.01, max(test_accs_list) + 0.01])
    plt.grid(True, axis='y', alpha=0.3)
    
    # add values on bars
    for i, v in enumerate(test_accs_list):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("knn_weights_comparison_plot.png", dpi=150, bbox_inches='tight')
 #   plt.show()
    
    return results, acc_diff, relative_diff

def test_metric_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    """
    Test sensitivity to metric (distance metric) hyperparameter
    metric = how to calculate distance between points
    - 'euclidean': straight-line distance (default)
      → d = sqrt((x1-x2)^2 + (y1-y2)^2)
      → most common, works well generally
    - 'manhattan': city-block distance (sum of absolute differences)
      → d = |x1-x2| + |y1-y2|
      → less sensitive to outliers than euclidean
    - 'minkowski': generalization (p=1 is manhattan, p=2 is euclidean)
      → with p=3, even less sensitive to outliers
    
    Choice depends on data characteristics and feature space
    """
    #removing minkowski as its taking forever!
    metrics = ['euclidean', 'manhattan']
    results = {}
    
    for metric in metrics:
        # minkowski with p=2 is same as euclidean, so use p=3
      #  if metric == 'minkowski':
           # model = KNeighborsClassifier(n_neighbors=k, metric=metric, p=3)
   #     else:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        results[metric] = {'train': train_acc, 'test': test_acc}
        print(f"{metric}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # calculate difference
    test_accs = [results[m]['test'] for m in metrics]
    acc_diff = max(test_accs) - min(test_accs)
    relative_diff = (acc_diff / min(test_accs)) * 100
    
    # bar plot
    plt.figure(figsize=(8, 5))
    test_accs_list = [results[m]['test'] for m in metrics]
    
    plt.bar(metrics, test_accs_list, color=['lightcoral', 'skyblue'])
    plt.ylabel("Test Accuracy")
    plt.title(f"Distance Metric Comparison (k={k})")
    plt.ylim([min(test_accs_list) - 0.01, max(test_accs_list) + 0.01])
    plt.grid(True, axis='y', alpha=0.3)
    
    # add values on bars
    for i, v in enumerate(test_accs_list):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("knn_metric_comparison_plot.png", dpi=150, bbox_inches='tight')
  #  plt.show()
    
    return results, acc_diff, relative_diff

"""
def test_algorithm_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    ""
    Test sensitivity to algorithm hyperparameter
    algorithm = how to compute nearest neighbors
    - 'auto': automatically picks best algorithm based on fit() data
    - 'ball_tree': uses ball tree structure → good for high dimensions
    - 'kd_tree': uses kd tree structure → fast for low dimensions
    - 'brute': brute force search → slow but always works
    
    Usually 'auto' is fine. This matters more for very large datasets
    For most cases, the accuracy should be identical, only speed differs
    ""
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    results = {}
    
    for algo in algorithms:
        model = KNeighborsClassifier(n_neighbors=k, algorithm=algo)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        results[algo] = {'train': train_acc, 'test': test_acc}
        print(f"{algo}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # calculate difference (should be very small or zero)
    test_accs = [results[a]['test'] for a in algorithms]
    acc_diff = max(test_accs) - min(test_accs)
    relative_diff = (acc_diff / min(test_accs)) * 100 if min(test_accs) > 0 else 0
    
    # bar plot
    plt.figure(figsize=(10, 5))
    test_accs_list = [results[a]['test'] for a in algorithms]
    
    plt.bar(algorithms, test_accs_list, color=['lightcoral', 'skyblue', 'lightgreen', 'orange'])
    plt.ylabel("Test Accuracy")
    plt.title(f"Algorithm Comparison (k={k})")
    plt.ylim([min(test_accs_list) - 0.01, max(test_accs_list) + 0.01])
    plt.grid(True, axis='y', alpha=0.3)
    
    # add values on bars
    for i, v in enumerate(test_accs_list):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("knn_algorithm_comparison_plot.png", dpi=150, bbox_inches='tight')
  #  plt.show()
    
    return results, acc_diff, relative_diff
"""
# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess
    X, Y = load_and_preprocess("../obesity.csv", "NObeyesdad")
    
    # split and scale (CRUCIAL for KNN!)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\nRunning n_neighbors (k) sensitivity test...")
    k_results, k_min, k_max, k_sensitivity = test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning weights comparison...")
    weights_results, weights_diff, weights_sensitivity = test_weights_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning metric comparison...")
    metric_results, metric_diff, metric_sensitivity = test_metric_sensitivity(X_train, X_test, Y_train, Y_test)
    
  #  print("\nRunning algorithm comparison...")
   # algo_results, algo_diff, algo_sensitivity = test_algorithm_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("knn_sensitivity_results.txt", "w") as f:
        f.write("KNN HYPERPARAMETER SENSITIVITY ANALYSIS - OBESITY DATASET\n\n")
        
        # TEST 1: n_neighbors
        f.write("TEST 1: n_neighbors (k) SENSITIVITY\n\n")
        f.write("Individual k value results:\n")
        for k, acc in k_results.items():
            f.write(f"k = {k}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nRelative sensitivity to k: {k_sensitivity:.4f}%\n")
        f.write(f"Min accuracy: {k_min:.4f}\n")
        f.write(f"Max accuracy: {k_max:.4f}\n")
        f.write(f"Plot saved as: knn_k_sensitivity_plot.png\n\n")
        
        # TEST 2: weights
        f.write("TEST 2: WEIGHTS COMPARISON (k=5)\n\n")
        for weight, acc in weights_results.items():
            f.write(f"{weight}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nAbsolute difference: {weights_diff:.4f}\n")
        f.write(f"Relative difference: {weights_sensitivity:.4f}%\n")
        f.write(f"Plot saved as: knn_weights_comparison_plot.png\n\n")
        
        # TEST 3: metric
        f.write("TEST 3: DISTANCE METRIC COMPARISON (k=5)\n\n")
        for metric, acc in metric_results.items():
            f.write(f"{metric}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nAbsolute difference: {metric_diff:.4f}\n")
        f.write(f"Relative difference: {metric_sensitivity:.4f}%\n")
        f.write(f"Plot saved as: knn_metric_comparison_plot.png\n\n")
        
        """
        # TEST 4: algorithm
        f.write("TEST 4: ALGORITHM COMPARISON (k=5)\n\n")
        for algo, acc in algo_results.items():
            f.write(f"{algo}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nAbsolute difference: {algo_diff:.4f}\n")
        f.write(f"Relative difference: {algo_sensitivity:.4f}%\n")
        f.write(f"Plot saved as: knn_algorithm_comparison_plot.png\n\n")
        """
        # SUMMARY
        f.write("SUMMARY\n")
        sensitivities = {
            'n_neighbors': k_sensitivity,
            'weights': weights_sensitivity,
            'metric': metric_sensitivity
        #    'algorithm': algo_sensitivity
        }
        most_sensitive = max(sensitivities, key=sensitivities.get)
        f.write(f"Most sensitive to: {most_sensitive} ({sensitivities[most_sensitive]:.2f}%)\n")
        f.write(f"\nSensitivity ranking:\n")
        for param, sens in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{param}: {sens:.2f}%\n")
    
    print("\nResults saved to: knn_sensitivity_results.txt")
    print("Plots saved as: knn_k_sensitivity_plot.png, knn_weights_comparison_plot.png, knn_metric_comparison_plot.png")