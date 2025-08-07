import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load and preprocess car data
df = pd.read_csv(".../Data.csv", encoding='cp1252')

df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')

# Encode categorical features
#categorical_features = ['brand', 'type', 'fuel']
categorical_features = ['MODELGROUP', 'TYPE', 'FUEL']
encoder = OneHotEncoder()
encoded_cats = encoder.fit_transform(df[categorical_features]).toarray()

# Normalize numerical features
scaler = MinMaxScaler()
#numerical_features = scaler.fit_transform(df[['price', 'horsepower']])
numerical_features = scaler.fit_transform(df[['PRICE']])

# Combine all features into car feature vectors
car_vectors = np.hstack((encoded_cats, numerical_features))

# Save car IDs for retrieval
car_ids = df['car_id'].values
car_info = df.set_index('car_id').to_dict(orient='index')


def recommend_cars(customer_input, top_n=5):
    # Prepare customer vector (same order of features as car vectors)
    cat_input = [[
        #customer_input['brand'],
        customer_input['model_group'],
        customer_input['type'],
        customer_input['fuel']
    ]]

    encoded_input = encoder.transform(cat_input).toarray()

    num_input = [[customer_input['price']]]
    normalized_input = scaler.transform(num_input)

    customer_vector = np.hstack((encoded_input, normalized_input))

    # Compute cosine similarity
    similarities = cosine_similarity(customer_vector, car_vectors)[0]

    # Get top N recommendations
    #top_indices = similarities.argsort()[::-1][:top_n]
    top_indices = similarities.argsort()[::-1]

    #new
    seen_modelgroups = set()
    recommendations = []

    for idx in top_indices:
        car_id = car_ids[idx]

        #new
        raw_data = car_info[car_id].copy()

        # Clean any non-serializable fields (optional)
        car_data = {
            k: v.isoformat() if isinstance(v, pd.Timestamp) else float(v) if isinstance(v,
                                                                                        (np.float32, np.float64)) else v
            for k, v in raw_data.items()
        }

        #car_data = car_info[car_id]
        car_data['score'] = round(similarities[idx], 2)

        #new
        modelgroup = car_data.get('MODELGROUP')

        #recommendations.append(car_data)

        # Avoid duplicates
        if modelgroup and modelgroup not in seen_modelgroups:
            seen_modelgroups.add(modelgroup)
            recommendations.append(car_data)

        if len(recommendations) == top_n:
            break

    return recommendations

