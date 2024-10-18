import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ecomerce_customer_data_cleaned.csv')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Convert non-numeric ages to NaN

# Define age ranges
age_ranges = {
    'Adults (18-39)': (18, 39),
    'Middle Age Adults (40-59)': (40, 59),
    'Older Adults (60-87)': (60, 87)
}

# Function to categorize ages into groups
def categorize_age(age):
    if pd.isna(age):
        return 'Other'
    elif 18 <= age <= 39:
        return 'Adults (18-39)'
    elif 40 <= age <= 59:
        return 'Middle Age Adults (40-59)'
    elif 60 <= age <= 87:
        return 'Older Adults (60-87)'
    else:
        return 'Other'

# Categorize age and calculate percentages
df['AgeGroup'] = df['Age'].apply(categorize_age)
age_group_counts = df['AgeGroup'].value_counts(normalize=True) * 100
print("\nAge Group Percentages:")
print(age_group_counts)


# Plot pie chart for age group percentages
plt.figure(figsize=(8, 6))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'skyblue', 'lightgreen'])
plt.title('Age Group Distribution in Population')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Find the most popular favorite and second-favorite categories by age group
def find_popular_categories(df, age_group):
    filtered_df = df[df['AgeGroup'] == age_group]

    if filtered_df.empty:
        return None, None

    # Get the most common favorite category and its count
    most_common_favorite = filtered_df['FavoriteCategory'].mode()[0] if not filtered_df['FavoriteCategory'].isna().all() else None
    count_most_common = filtered_df['FavoriteCategory'].value_counts().get(most_common_favorite, 0)

    return most_common_favorite, count_most_common


# Display popular categories by age group
favorite_categories = {}
category_counts = {}
for age_group in age_ranges.keys():
    fav_category, count = find_popular_categories(df, age_group)

    if fav_category is None:
        print(f"\nAge Group: {age_group}")
        print("No customers in this age group.")
    else:
        print(f"\nAge Group: {age_group}")
        print(f"Most Popular Favorite Category: {fav_category}")
        print(f"Count: {count}")
        favorite_categories[age_group] = fav_category
        category_counts[age_group] = count

# Plotting the age group percentages

print("\nFavorite Categories by Age Group:")
print(favorite_categories)


# Plotting the most favorite category counts by age group
age_groups = list(favorite_categories.keys())
categories = list(favorite_categories.values())
counts = list(category_counts.values())

plt.figure(figsize=(8, 6))
bars = plt.bar(age_groups, counts, color='skyblue')

# Add category labels on top of the bars
for bar, category in zip(bars, categories):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, category, ha='center', va='bottom')

plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Most Favorite Category by Age Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Convert 'AverageOrderValue' and 'TotalPurchases' to numeric
df['AverageOrderValue'] = pd.to_numeric(df['AverageOrderValue'], errors='coerce')
df['TotalPurchases'] = pd.to_numeric(df['TotalPurchases'], errors='coerce')

# Drop rows with missing values in the relevant columns
df.dropna(subset=['AverageOrderValue', 'TotalPurchases'], inplace=True)

# Create a feature matrix for categories by age group
category_features_by_age_group = df.groupby(['AgeGroup', 'FavoriteCategory']).agg({
    'AverageOrderValue': 'mean',
    'TotalPurchases': 'mean'
}).reset_index()

print("\nCategory Features by Age Group:")
print(category_features_by_age_group)

# Function to calculate similarity based on Euclidean distance for a specific age group
def calculate_similarity_for_age_group(target_category, age_group, feature_matrix):
    age_group_feature_matrix = feature_matrix[feature_matrix['AgeGroup'] == age_group].set_index('FavoriteCategory')

    # Ensure the target category exists
    if target_category not in age_group_feature_matrix.index:
        return f"{target_category} not found in {age_group} category data."

    target_vector = age_group_feature_matrix.loc[target_category, ['AverageOrderValue', 'TotalPurchases']].values # Select only numeric columns
    distances = {}

    # Calculate Euclidean distance to other categories
    for category, vector in age_group_feature_matrix.iterrows():
        if category != target_category:
            distances[category] = euclidean(target_vector, vector[['AverageOrderValue', 'TotalPurchases']].values) # Select only numeric columns

    # Sort and return top 10 categories by similarity
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    return sorted_distances[:10]

# Calculate and display top 10 similar categories for each age group
for age_group, category in favorite_categories.items():
    top_similar_categories = calculate_similarity_for_age_group(category, age_group, category_features_by_age_group)
    print(f"\nTop 10 similar categories to {category} for {age_group}:")
    if isinstance(top_similar_categories, str):  # Error message
        print(top_similar_categories)
    else:
        for cat, dist in top_similar_categories:
            print(f"Category: {cat}, Distance: {dist:.4f}")


