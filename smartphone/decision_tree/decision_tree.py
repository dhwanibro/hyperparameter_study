import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# ### The target variable is "Activity"
# 1 = WALKING
# 2 = WALKING_UPSTAIRS
# 3 = WALKING_DOWNSTAIRS
# 4 = SITTING
# 5 = STANDING
# 6 = LAYING

def load_and_preprocess(train_path, test_path, target_col):
    """Load data and do preprocessing"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # check for missing values in train
    missing_train = train_df.isnull().sum()
    if missing_train.sum() > 0:
        print(f"Missing values in train:\n{missing_train[missing_train > 0]}")
        train_df = train_df.dropna()
    else:
        print("No missing values in train")
    
    # check for missing values in test
    missing_test = test_df.isnull().sum()
    if missing_test.sum() > 0:
        print(f"Missing values in test:\n{missing_test[missing_test > 0]}")
        test_df = test_df.dropna()
    else:
        print("No missing values in test")
    
    # check for duplicates in train
    duplicates_train = train_df.duplicated().sum()
    if duplicates_train > 0:
        print(f"Found {duplicates_train} duplicate rows in train, removing...")
        train_df = train_df.drop_duplicates()
    
    # check for duplicates in test
    duplicates_test = test_df.duplicated().sum()
    if duplicates_test > 0:
        print(f"Found {duplicates_test} duplicate rows in test, removing...")
        test_df = test_df.drop_duplicates()
    
    # separate features and target for train
    X_train = train_df.drop(target_col, axis=1)
    Y_train = train_df[target_col]
    
    # separate features and target for test
    X_test = test_df.drop(target_col, axis=1)
    Y_test = test_df[target_col]
    
    # check class balance
    print(f"\nTrain class distribution:")
    unique, counts = np.unique(Y_train, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y_train)*100:.2f}%)")
    
    print(f"\nTest class distribution:")
    unique, counts = np.unique(Y_test, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y_test)*100:.2f}%)")
    
    return X_train, X_test, Y_train, Y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # creating an instance of the standardscaler class
    # scales features to mean=0, std=1 so extreme values don't distort training
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    # the scaler object stores the mean and std of every column inside itself. transform just applies it
    
    # applying the train data's mean and sd to this, will not fit it cause that causes leakage
    X_test = scaler.transform(X_test)
    
    return X_train, X_test
def test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test, depth_values=[1, 2, 3, 5, 7, 10, 15, 20, None]):
    """
    Test sensitivity to max_depth hyperparameter
    max_depth = maximum depth of the tree
    - Low depth (1-3) → shallow tree → underfitting → high bias
    - High depth (20+) → deep tree → overfitting → high variance
    - None → tree grows until all leaves are pure (usually overfits)
    
    Depth controls model complexity
    """
    test_acc_list = []
    depth_results = {}
    
    for depth in depth_values:
        # Create decision tree classifier
        # random_state=42 makes results reproducible
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, Y_train)  # fit takes (features, labels)
        
        # predict() takes FEATURES (X), not labels (Y)
        Y_train_pred = model.predict(X_train)  # predict on training features
        Y_test_pred = model.predict(X_test)    # predict on test features
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        test_acc_list.append(test_acc)
        depth_results[depth] = {'train': train_acc, 'test': test_acc}
        
        print(f"Depth={depth}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    min_acc = min(test_acc_list)
    max_acc = max(test_acc_list)
    
    relative_change = (max_acc - min_acc) / min_acc
    percent_change = relative_change * 100
    
    print(f"\nmax_depth Sensitivity: {percent_change:.4f}%")
    
    # plot
    plt.figure(figsize=(10, 6))
    # Convert None to string for plotting
    x_labels = [str(d) for d in depth_values]
    x_positions = range(len(depth_values))
    
    plt.plot(x_positions, test_acc_list, marker="o", label="Test Accuracy")
    plt.xticks(x_positions, x_labels)
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity to max_depth")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_depth_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return depth_results, min_acc, max_acc, percent_change

def test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test, split_values=[2, 5, 10, 20, 50, 100]):
    """
    Test sensitivity to min_samples_split hyperparameter
    min_samples_split = minimum number of samples required to split an internal node
    - Low values (2-5) → allows more splits → complex tree → can overfit
    - High values (50+) → fewer splits → simpler tree → can underfit
    
    This controls how "eager" the tree is to split nodes
    """
    test_acc_list = []
    split_results = {}
    
    for split in split_values:
        # min_samples_split must be at least 2
        model = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        test_acc_list.append(test_acc)
        split_results[split] = {'train': train_acc, 'test': test_acc}
        
        print(f"min_samples_split={split}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    min_acc = min(test_acc_list)
    max_acc = max(test_acc_list)
    
    relative_change = (max_acc - min_acc) / min_acc
    percent_change = relative_change * 100
    
    print(f"\nmin_samples_split Sensitivity: {percent_change:.4f}%")
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(split_values, test_acc_list, marker="o", label="Test Accuracy")
    plt.xlabel("min_samples_split")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity to min_samples_split")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_split_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return split_results, min_acc, max_acc, percent_change

def test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test, leaf_values=[1, 2, 5, 10, 20, 50]):
    """
    Test sensitivity to min_samples_leaf hyperparameter
    min_samples_leaf = minimum number of samples required to be at a leaf node
    - Low values (1-2) → allows tiny leaves → can overfit to noise
    - High values (20+) → forces bigger leaves → smoother decision boundary
    
    This is another way to prevent overfitting
    """
    test_acc_list = []
    leaf_results = {}
    
    for leaf in leaf_values:
        model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        test_acc_list.append(test_acc)
        leaf_results[leaf] = {'train': train_acc, 'test': test_acc}
        
        print(f"min_samples_leaf={leaf}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    min_acc = min(test_acc_list)
    max_acc = max(test_acc_list)
    
    relative_change = (max_acc - min_acc) / min_acc
    percent_change = relative_change * 100
    
    print(f"\nmin_samples_leaf Sensitivity: {percent_change:.4f}%")
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_values, test_acc_list, marker="o", label="Test Accuracy")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity to min_samples_leaf")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_leaf_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return leaf_results, min_acc, max_acc, percent_change

def test_criterion_sensitivity(X_train, X_test, Y_train, Y_test):
    """
    Test sensitivity to criterion (splitting criterion)
    criterion = the function to measure split quality
    - 'gini': Gini impurity (default) → measures how often a random sample would be misclassified
    - 'entropy': Information gain → measures information disorder
    - 'log_loss': Log loss, also known as cross-entropy
    
    Usually gini and entropy give similar results
    Gini is faster to compute, entropy might be slightly more accurate
    """
    criteria = ['gini', 'entropy', 'log_loss']
    results = {}
    
    for crit in criteria:
        model = DecisionTreeClassifier(criterion=crit, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        results[crit] = {'train': train_acc, 'test': test_acc}
        print(f"{crit}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # calculate difference between best and worst
    test_accs = [results[c]['test'] for c in criteria]
    acc_diff = max(test_accs) - min(test_accs)
    relative_diff = (acc_diff / min(test_accs)) * 100
    
    # bar plot
    plt.figure(figsize=(8, 5))
    test_accs_list = [results[c]['test'] for c in criteria]
    
    plt.bar(criteria, test_accs_list, color=['lightcoral', 'skyblue', 'lightgreen'])
    plt.ylabel("Test Accuracy")
    plt.title("Criterion Comparison")
    plt.ylim([min(test_accs_list) - 0.01, max(test_accs_list) + 0.01])
    plt.grid(True, axis='y', alpha=0.3)
    
    # add values on bars
    for i, v in enumerate(test_accs_list):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("dt_criterion_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, acc_diff, relative_diff

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess using existing train/test split
    X_train, X_test, Y_train, Y_test = load_and_preprocess(
        "../smartphone_train.csv", 
        "../smartphone_test.csv", 
        "Activity"
    )
    
    # scale features
    X_train, X_test = scale_features(X_train, X_test)
    # run tests
    print("\nRunning max_depth sensitivity test...")
    depth_results, depth_min, depth_max, depth_sensitivity = test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning min_samples_split sensitivity test...")
    split_results, split_min, split_max, split_sensitivity = test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning min_samples_leaf sensitivity test...")
    leaf_results, leaf_min, leaf_max, leaf_sensitivity = test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning criterion comparison...")
    criterion_results, criterion_diff, criterion_sensitivity = test_criterion_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("dt_sensitivity_results.txt", "w") as f:
        f.write("DECISION TREE HYPERPARAMETER SENSITIVITY ANALYSIS - HAR DATASET\n\n")
        
        # TEST 1: max_depth
        f.write("TEST 1: max_depth SENSITIVITY\n\n")
        f.write("Individual max_depth results:\n")
        for depth, acc in depth_results.items():
            f.write(f"max_depth = {depth}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nRelative sensitivity to max_depth: {depth_sensitivity:.4f}%\n")
        f.write(f"Min accuracy: {depth_min:.4f}\n")
        f.write(f"Max accuracy: {depth_max:.4f}\n")
        f.write(f"Plot saved as: dt_depth_sensitivity_plot.png\n\n")
        
        # TEST 2: min_samples_split
        f.write("TEST 2: min_samples_split SENSITIVITY\n\n")
        f.write("Individual min_samples_split results:\n")
        for split, acc in split_results.items():
            f.write(f"min_samples_split = {split}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nRelative sensitivity to min_samples_split: {split_sensitivity:.4f}%\n")
        f.write(f"Min accuracy: {split_min:.4f}\n")
        f.write(f"Max accuracy: {split_max:.4f}\n")
        f.write(f"Plot saved as: dt_split_sensitivity_plot.png\n\n")
        
        # TEST 3: min_samples_leaf
        f.write("TEST 3: min_samples_leaf SENSITIVITY\n\n")
        f.write("Individual min_samples_leaf results:\n")
        for leaf, acc in leaf_results.items():
            f.write(f"min_samples_leaf = {leaf}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nRelative sensitivity to min_samples_leaf: {leaf_sensitivity:.4f}%\n")
        f.write(f"Min accuracy: {leaf_min:.4f}\n")
        f.write(f"Max accuracy: {leaf_max:.4f}\n")
        f.write(f"Plot saved as: dt_leaf_sensitivity_plot.png\n\n")
        
        # TEST 4: criterion
        f.write("TEST 4: CRITERION COMPARISON\n\n")
        for crit, acc in criterion_results.items():
            f.write(f"{crit}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nAbsolute difference: {criterion_diff:.4f}\n")
        f.write(f"Relative difference: {criterion_sensitivity:.4f}%\n")
        f.write(f"Plot saved as: dt_criterion_comparison_plot.png\n\n")
        
        # SUMMARY
        f.write("SUMMARY\n")
        sensitivities = {
            'max_depth': depth_sensitivity,
            'min_samples_split': split_sensitivity,
            'min_samples_leaf': leaf_sensitivity,
            'criterion': criterion_sensitivity
        }
        most_sensitive = max(sensitivities, key=sensitivities.get)
        f.write(f"Most sensitive to: {most_sensitive} ({sensitivities[most_sensitive]:.2f}%)\n")
        f.write(f"\nSensitivity ranking:\n")
        for param, sens in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{param}: {sens:.2f}%\n")
    
    print("\nResults saved to: dt_sensitivity_results.txt")
    print("Plots saved as: dt_depth_sensitivity_plot.png, dt_split_sensitivity_plot.png, dt_leaf_sensitivity_plot.png, dt_criterion_comparison_plot.png")