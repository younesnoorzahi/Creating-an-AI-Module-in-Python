from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(data_path, target_column, test_size=0.2):
    """
    Train a simple Random Forest classifier
    
    Args:
        data_path (str): Path to CSV data file
        target_column (str): Name of target column
        test_size (float): Proportion of test data
        
    Returns:
        tuple: (model, test_accuracy)
    """
    data = pd.read_csv(data_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def predict(model, input_data):
    """
    Make predictions using trained model
    
    Args:
        model: Trained model
        input_data: Data for prediction
        
    Returns:
        array: Predictions
    """
    return model.predict(input_data)