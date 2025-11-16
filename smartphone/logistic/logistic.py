import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, C_values=[0.001, 0.01, 0.1, 1, 10, 100]):
    """
    Test sensitivity to regularization strength C
    C = 1/lambda (inverse regularization strength)
    low C → strong penalty → simpler model
    high C → weak penalty → model fits more → maybe overfit
    """
    test_acc_list = []
    c_results = {}  # store results for each C value
    
    for C in C_values:
        # multi_class='multinomial' for 6 activity classes
        model = LogisticRegression(C=C, max_iter=2000, multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        test_acc_list.append(test_acc)
        c_results[C] = {'train': train_acc, 'test': test_acc}
    
    min_acc = min(test_acc_list)
    max_acc = max(test_acc_list)
    
    relative_change = (max_acc - min_acc) / min_acc
    percent_change = relative_change * 100
    
    print(f"\nC Sensitivity: {percent_change:.4f}%")
    
    # plot and save
    plt.figure()
    plt.plot(C_values, test_acc_list, marker="o")
    
    # Because C jumps like: 0.001 → 0.01 → 0.1 → 1 → 10 → 100
    # A log scale spreads them evenly.
    plt.xscale("log")
    
    plt.xlabel("C value")
    plt.ylabel("Test Accuracy")
    plt.title("Sensitivity of Logistic Regression to C (HAR)")
    plt.grid(True)
    plt.savefig("har_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return c_results, min_acc, max_acc, percent_change

def test_penalty_types(X_train, X_test, Y_train, Y_test, C_fixed=1.0):
    """
    Test L1 vs L2 penalty with fixed C
    L1 (lasso) → can zero out features, does feature selection
    L2 (ridge) → shrinks all weights, keeps all features
    """
    penalties = ['l1', 'l2']
    results = {}
    
    for penalty in penalties:
        # liblinear works for both l1 and l2
        model = LogisticRegression(penalty=penalty, C=C_fixed, max_iter=2000, solver='liblinear')
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(Y_train, Y_train_pred)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        
        results[penalty] = {'train': train_acc, 'test': test_acc}
        print(f"{penalty.upper()} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # calculate difference in test accuracy between L1 and L2
    acc_diff = abs(results['l1']['test'] - results['l2']['test'])
    relative_diff = (acc_diff / min(results['l1']['test'], results['l2']['test'])) * 100
    
    # bar plot comparing L1 vs L2
    plt.figure(figsize=(8, 5))
    penalties_plot = ['L1', 'L2']
    test_accs = [results['l1']['test'], results['l2']['test']]
    
    plt.bar(penalties_plot, test_accs, color=['lightcoral', 'skyblue'])
    plt.ylabel("Test Accuracy")
    plt.title(f"L1 vs L2 Regularization (C={C_fixed}) - HAR")
    plt.ylim([min(test_accs) - 0.01, max(test_accs) + 0.01])
    plt.grid(True, axis='y', alpha=0.3)
    
    # add values on top of bars
    for i, v in enumerate(test_accs):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("har_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
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
    print("\nRunning C sensitivity test...")
    c_results, min_acc, max_acc, c_percent_change = test_C_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning L1 vs L2 penalty test...")
    penalty_results, acc_diff, relative_diff = test_penalty_types(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("har_sensitivity_results.txt", "w") as f:
        f.write("LOGISTIC REGRESSION HYPERPARAMETER SENSITIVITY ANALYSIS - HAR DATASET\n\n")
        
        f.write("TEST 1: C SENSITIVITY (Regularization Strength)\n\n")
        f.write("Individual C value results:\n")
        for C, acc in c_results.items():
            f.write(f"C = {C}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nRelative sensitivity to C: {c_percent_change:.4f}%\n")
        f.write(f"Min accuracy: {min_acc:.4f}\n")
        f.write(f"Max accuracy: {max_acc:.4f}\n")
        f.write(f"Plot saved as: har_c_sensitivity_plot.png\n\n")
        
        f.write("TEST 2: L1 vs L2 PENALTY (with C=1.0)\n")
        f.write(f"L1 Train Accuracy: {penalty_results['l1']['train']:.4f}\n")
        f.write(f"L1 Test Accuracy: {penalty_results['l1']['test']:.4f}\n")
        f.write(f"L2 Train Accuracy: {penalty_results['l2']['train']:.4f}\n")
        f.write(f"L2 Test Accuracy: {penalty_results['l2']['test']:.4f}\n")
        f.write(f"Absolute difference: {acc_diff:.4f}\n")
        f.write(f"Relative difference: {relative_diff:.4f}%\n")
        f.write(f"Plot saved as: har_penalty_comparison_plot.png\n\n")
        
        f.write("SUMMARY\n")
        f.write(f"More sensitive to: ")
        if c_percent_change > relative_diff:
            f.write(f"C values ({c_percent_change:.2f}% vs {relative_diff:.2f}%)\n")
        else:
            f.write(f"Penalty type ({relative_diff:.2f}% vs {c_percent_change:.2f}%)\n")
    
    print("\nResults saved to: har_sensitivity_results.txt")
    print("Plots saved as: har_c_sensitivity_plot.png, har_penalty_comparison_plot.png")