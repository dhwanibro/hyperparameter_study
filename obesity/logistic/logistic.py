import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
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
        # multi_class='multinomial' for multiclass obesity classification
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
    plt.title("Sensitivity of Logistic Regression to C (Obesity)")
    plt.grid(True)
    plt.savefig("obesity_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
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
    plt.title(f"L1 vs L2 Regularization (C={C_fixed}) - Obesity")
    plt.ylim([min(test_accs) - 0.01, max(test_accs) + 0.01])
    plt.grid(True, axis='y', alpha=0.3)
    
    # add values on top of bars
    for i, v in enumerate(test_accs):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("obesity_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, acc_diff, relative_diff


if __name__ == "__main__":
    # load and preprocess
    X, Y = load_and_preprocess("../obesity.csv", "NObeyesdad")
    
    # split and scale
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\nRunning C sensitivity test...")
    c_results, min_acc, max_acc, c_percent_change = test_C_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\nRunning L1 vs L2 penalty test...")
    penalty_results, acc_diff, relative_diff = test_penalty_types(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("obesity_sensitivity_results.txt", "w") as f:
        f.write("LOGISTIC REGRESSION HYPERPARAMETER SENSITIVITY ANALYSIS - OBESITY DATASET\n\n")
        
        f.write("TEST 1: C SENSITIVITY (Regularization Strength)\n\n")
        f.write("Individual C value results:\n")
        for C, acc in c_results.items():
            f.write(f"C = {C}: Train Acc = {acc['train']:.4f}, Test Acc = {acc['test']:.4f}\n")
        f.write(f"\nRelative sensitivity to C: {c_percent_change:.4f}%\n")
        f.write(f"Min accuracy: {min_acc:.4f}\n")
        f.write(f"Max accuracy: {max_acc:.4f}\n")
        f.write(f"Plot saved as: obesity_c_sensitivity_plot.png\n\n")
        
        f.write("TEST 2: L1 vs L2 PENALTY (with C=1.0)\n")
        f.write(f"L1 Train Accuracy: {penalty_results['l1']['train']:.4f}\n")
        f.write(f"L1 Test Accuracy: {penalty_results['l1']['test']:.4f}\n")
        f.write(f"L2 Train Accuracy: {penalty_results['l2']['train']:.4f}\n")
        f.write(f"L2 Test Accuracy: {penalty_results['l2']['test']:.4f}\n")
        f.write(f"Absolute difference: {acc_diff:.4f}\n")
        f.write(f"Relative difference: {relative_diff:.4f}%\n")
        f.write(f"Plot saved as: obesity_penalty_comparison_plot.png\n\n")
        
        f.write("SUMMARY\n")
        f.write(f"More sensitive to: ")
        if c_percent_change > relative_diff:
            f.write(f"C values ({c_percent_change:.2f}% vs {relative_diff:.2f}%)\n")
        else:
            f.write(f"Penalty type ({relative_diff:.2f}% vs {c_percent_change:.2f}%)\n")
    
    print("\nResults saved to: obesity_sensitivity_results.txt")
    print("Plots saved as: obesity_c_sensitivity_plot.png, obesity_penalty_comparison_plot.png")