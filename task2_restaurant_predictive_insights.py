import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib.util
import sys

class RestaurantPredictiveInsights:
    def __init__(self, zomato_dataframe):
        self.original_data = zomato_dataframe
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.output_dir = 'processed_data/predictive_insights'
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_columns = []  # Track the actual feature columns used
        self.num_price_categories = 4  # Default, will be updated during preprocessing
        self.num_cuisine_categories = 0  # Will be set during preprocessing

    def preprocess_data(self):
        df = self.original_data.copy()
        print("Available columns:", df.columns.tolist())

        # Map the actual column names to expected names
        column_mapping = {
            'approx_cost(for two people)': 'Average Cost for two',
            'votes': 'Votes',
            'rate': 'Aggregate rating',
            # No direct mapping for Price range, will create it
        }

        # Create mapped columns where possible
        for original, expected in column_mapping.items():
            if original in df.columns:
                df[expected] = df[original]
                print(f"Mapped '{original}' to '{expected}'")
            else:
                print(f"Warning: Original column '{original}' not found")
                df[expected] = 0  # Default value

        # Handle aggregate rating
        if 'Aggregate rating' in df.columns:
            # Convert ratings to numeric, handling any non-numeric values
            if df['Aggregate rating'].dtype == object:  # If it's a string
                df['Aggregate rating'] = df['Aggregate rating'].str.replace('/5', '').str.strip()
            df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
            df['Aggregate rating'] = df['Aggregate rating'].fillna(0)
            print("Processed 'Aggregate rating'")

        # Create price range from cost if not present
        if 'Price range' not in df.columns:
            if 'Average Cost for two' in df.columns:
                # Convert to numeric, handling any non-numeric values
                df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')
                df['Average Cost for two'] = df['Average Cost for two'].fillna(0)
                
                # Create price range (0-3) based on quartiles of cost
                # Important: Using 0-3 range to ensure compatibility with sparse_categorical_crossentropy
                if df['Average Cost for two'].nunique() > 3:
                    df['Price range'] = pd.qcut(
                        df['Average Cost for two'],
                        q=4,
                        labels=[0, 1, 2, 3]
                    ).astype(int)
                else:
                    # If not enough unique values, create simpler categories
                    df['Price range'] = pd.cut(
                        df['Average Cost for two'],
                        bins=[0, 500, 1000, float('inf')],
                        labels=[0, 1, 2]
                    ).fillna(0).astype(int)
                
                # Get the actual number of unique categories
                self.num_price_categories = len(np.unique(df['Price range']))
                print(f"Created 'Price range' from 'Average Cost for two', found {self.num_price_categories} unique categories: {np.unique(df['Price range'])}")
            else:
                df['Price range'] = 0  # Default value
                self.num_price_categories = 1
                print("Created default 'Price range'")

        # Handle cuisine data
        if 'listed_in(type)' in df.columns and 'Cuisines' not in df.columns:
            # Use restaurant type as a proxy for cuisine if actual cuisines aren't available
            df['Primary Cuisine'] = df['listed_in(type)']
            print("Using 'listed_in(type)' as proxy for 'Primary Cuisine'")
        else:
            print("Warning: No cuisine data found, using 'Unknown'")
            df['Primary Cuisine'] = 'Unknown'

        # Ensure all necessary features exist
        required_features = ['Average Cost for two', 'Votes', 'Aggregate rating', 'Price range']
        self.feature_columns = []
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
                print(f"Created missing column '{feature}' with default values")
            self.feature_columns.append(feature)
        
        # Encode cuisine
        self.encoders['cuisine'] = LabelEncoder()
        df['Cuisine_Encoded'] = self.encoders['cuisine'].fit_transform(df['Primary Cuisine'])
        self.feature_columns.append('Cuisine_Encoded')
        self.num_cuisine_categories = len(self.encoders['cuisine'].classes_)
        print(f"Found {self.num_cuisine_categories} unique cuisine categories")

        # Store the final feature list for later use
        print(f"Final feature columns: {self.feature_columns}")
        
        X = df[self.feature_columns]

        prediction_targets = {
            'rating_prediction': df['Aggregate rating'],
            'price_range_prediction': df['Price range'],
            'cuisine_prediction': df['Cuisine_Encoded']
        }

        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)

        self.processed_data = {'features': X_scaled, 'targets': prediction_targets}
        return X_scaled, prediction_targets

    def build_multi_task_model(self, input_shape):
        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)

        # Rating prediction (regression)
        rating_output = Dense(1, name='rating_prediction')(x)
        
        # Price range prediction (classification)
        # Ensure at least 4 outputs or the actual number of categories, whichever is greater
        price_categories = max(4, self.num_price_categories)
        price_range_output = Dense(price_categories, activation='softmax', name='price_range_prediction')(x)
        
        # Cuisine prediction (classification)
        cuisine_output = Dense(self.num_cuisine_categories, activation='softmax', name='cuisine_prediction')(x)

        model = Model(inputs=inputs, outputs=[rating_output, price_range_output, cuisine_output])
        
        model.compile(
            optimizer='adam',
            loss={
                'rating_prediction': 'mse',
                'price_range_prediction': 'sparse_categorical_crossentropy',
                'cuisine_prediction': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'rating_prediction': 1.0,
                'price_range_prediction': 0.8,
                'cuisine_prediction': 0.5
            },
            metrics={
                'rating_prediction': ['mae'],
                'price_range_prediction': ['accuracy'],
                'cuisine_prediction': ['accuracy']
            }
        )
        
        print(f"Model output shapes: rating (1), price range ({price_categories}), cuisine ({self.num_cuisine_categories})")
        return model

    def train_predictive_model(self):
        X, targets = self.preprocess_data()
        X_train, X_test, train_idx, test_idx = train_test_split(X, np.arange(len(X)), test_size=0.2, random_state=42)

        y_rating_train = targets['rating_prediction'].iloc[train_idx].values
        y_price_train = targets['price_range_prediction'].iloc[train_idx].values
        y_cuisine_train = targets['cuisine_prediction'].iloc[train_idx].values

        y_rating_test = targets['rating_prediction'].iloc[test_idx].values  
        y_price_test = targets['price_range_prediction'].iloc[test_idx].values
        y_cuisine_test = targets['cuisine_prediction'].iloc[test_idx].values

        # Print the unique values and check for issues BEFORE building the model
        unique_price_values = np.unique(y_price_train)
        unique_cuisine_values = np.unique(y_cuisine_train)
        print(f"Unique price range values in training set: {unique_price_values}")
        print(f"Unique cuisine values in training set: {unique_cuisine_values}")
    
        # Update category counts if the detected values exceed what we expected
        max_price_value = int(np.max(y_price_train))
        if max_price_value >= self.num_price_categories:
            self.num_price_categories = max_price_value + 1
            print(f"IMPORTANT: Adjusted price categories to {self.num_price_categories} based on actual data")
    
        max_cuisine_value = int(np.max(y_cuisine_train))
        if max_cuisine_value >= self.num_cuisine_categories:
            self.num_cuisine_categories = max_cuisine_value + 1
            print(f"IMPORTANT: Adjusted cuisine categories to {self.num_cuisine_categories} based on actual data")

        # Now build the model with the correct output sizes
        model = self.build_multi_task_model(X_train.shape[1])
        self.models['multi_task'] = model

        # Convert target arrays to the correct shape
        y_rating_train = np.expand_dims(y_rating_train, axis=1)  # Shape: (n_samples, 1)
    
        # Get output sizes from layer configs instead of output shape
        price_output_layer = model.get_layer('price_range_prediction')
        cuisine_output_layer = model.get_layer('cuisine_prediction')
        price_output_size = price_output_layer.get_config()['units']
        cuisine_output_size = cuisine_output_layer.get_config()['units']
    
        # Double-check that price and cuisine labels are within model output range
        if max_price_value >= price_output_size:
            raise ValueError(f"Price range values (max={max_price_value}) exceed model output size ({price_output_size})")
    
        if max_cuisine_value >= cuisine_output_size:
            raise ValueError(f"Cuisine values (max={max_cuisine_value}) exceed model output size ({cuisine_output_size})")
    
        print("All label values are within valid output ranges. Proceeding with training...")

        history = model.fit(
            X_train,
            {
                'rating_prediction': y_rating_train,
                'price_range_prediction': y_price_train,  # Keep as is for sparse_categorical_crossentropy
                'cuisine_prediction': y_cuisine_train     # Keep as is for sparse_categorical_crossentropy
            },
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        self._visualize_predictions(model, X_test, {
            'rating_prediction': y_rating_test,
            'price_range_prediction': y_price_test,
            'cuisine_prediction': y_cuisine_test
        })

        return model, history

    def _visualize_predictions(self, model, X_test, y_test):
        predictions = model.predict(X_test)

        plt.figure(figsize=(15, 5))

        # Rating predictions (regression)
        plt.subplot(131)
        plt.scatter(y_test['rating_prediction'], predictions[0].flatten())
        plt.title('Actual vs Predicted Ratings')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')

        # Price range predictions (classification)
        plt.subplot(132)
        predicted_price_classes = np.argmax(predictions[1], axis=1)
        confusion_price = tf.math.confusion_matrix(
            y_test['price_range_prediction'],
            predicted_price_classes
        )
        sns.heatmap(confusion_price, annot=True, fmt='d', cmap='Blues')
        plt.title('Price Range Prediction Confusion Matrix')

        # Cuisine predictions (classification)
        plt.subplot(133)
        predicted_cuisine_classes = np.argmax(predictions[2], axis=1)
        confusion_cuisine = tf.math.confusion_matrix(
            y_test['cuisine_prediction'],
            predicted_cuisine_classes
        )
        sns.heatmap(confusion_cuisine, annot=True, fmt='d', cmap='Blues')
        plt.title('Cuisine Prediction Confusion Matrix')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_visualizations.png'))
        plt.close()

    def predict_restaurant_insights(self, new_restaurant_data):
        # Ensure all expected features are present in the correct order
        for col in self.feature_columns:
            if col not in new_restaurant_data.columns:
                new_restaurant_data[col] = 0
                print(f"Added missing column '{col}' with default value 0")
        
        # Ensure only the expected columns are used, in the correct order
        input_data = new_restaurant_data[self.feature_columns].copy()
        print(f"Input data shape: {input_data.shape}")
        print(f"Input data columns: {input_data.columns.tolist()}")
        
        # Scale the data using the previously fit scaler
        scaled_data = self.scalers['features'].transform(input_data)
        
        # Make predictions
        predictions = self.models['multi_task'].predict(scaled_data)
        
        # Process predictions based on the model architecture
        predicted_price_class = np.argmax(predictions[1], axis=1)[0]
        predicted_cuisine_class = np.argmax(predictions[2], axis=1)[0]
        
        # Return formatted predictions
        return {
            'predicted_rating': float(predictions[0][0][0]),
            'predicted_price_range': int(predicted_price_class),
            'predicted_cuisine': self.encoders['cuisine'].inverse_transform([predicted_cuisine_class])[0]
        }

def main():
    module_path = "e:/DATA SCIENCE/CODTECH/task1_zomato_etl_pipeline_simplified.py"
    spec = importlib.util.spec_from_file_location("etl_module", module_path)
    etl_module = importlib.util.module_from_spec(spec)
    sys.modules["etl_module"] = etl_module
    spec.loader.exec_module(etl_module)
    run_etl_pipeline = etl_module.run_etl_pipeline

    zomato_data = run_etl_pipeline('Zomato-data-.csv')
    predictive_insights = RestaurantPredictiveInsights(zomato_data)
    model, history = predictive_insights.train_predictive_model()

    # Use the same column names that were used during training
    new_restaurant_data = pd.DataFrame({
        col: [0] for col in predictive_insights.feature_columns
    })
    
    # Set some example values
    if 'Average Cost for two' in new_restaurant_data.columns:
        new_restaurant_data['Average Cost for two'] = [1000]
    if 'Votes' in new_restaurant_data.columns:
        new_restaurant_data['Votes'] = [500]
    
    print("New restaurant data columns:", new_restaurant_data.columns.tolist())
    insights = predictive_insights.predict_restaurant_insights(new_restaurant_data)
    print("Predictive Insights:", insights)

if __name__ == "__main__":
    main()