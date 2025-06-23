import matplotlib.pyplot as plt
import joblib

def visualize_result(y_test, y_pred, save_path='results.png'):
    """
    Plots actual vs predicted values and saves the figure.
    
    Args:
        y_test (dask.Series or DataFrame): The true labels (or target values).
        y_pred (dask.Series or DataFrame): The predicted labels or values.
        save_path (str): Path for saving the plot image.
    """
    plt.figure(figsize=(10,6))
    plt.scatter(y_test.compute()[:1000], y_pred.compute()[:1000], alpha=0.3)
    plt.plot([0, max(y_test.compute()[:1000])], [0, max(y_test.compute()[:1000])], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Trip Duration')
    plt.savefig(save_path)
    plt.close()

def save_preprocess_data(X_train, y_train, X_test, y_test, filename_prefix='preprocessed'):
    """
    Save the preprocessed training and testing data as pickle files.
    
    Args:
        X_train, y_train, X_test, y_test: Data to be saved.
        filename_prefix (str): Common prefix for all filenames.
    """
    import os
    import pickle
    
    os.makedirs('preprocessed_data', exist_ok=True)
    with open(f'preprocessed_data/{filename_prefix}_X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(f'preprocessed_data/{filename_prefix}_y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(f'preprocessed_data/{filename_prefix}_X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(f'preprocessed_data/{filename_prefix}_y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

def save_trained_model(model, model_path, filename='trained_model.pkl'):
    """
    Save trained scikit-learn model to disk.
    
    Args:
        model: Trained model object.
        filename (str): Filename for saving the model.
    """
    import os
    if os.path.exists(model_path):
        print("File already exists")
    else:
        os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, filename))