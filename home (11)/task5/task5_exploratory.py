import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency, f_oneway, spearmanr, pearsonr
from task5_main import preprocess_data
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Load preprocessed data
df = preprocess_data()

# Create a figure directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Focus features
focus_features = [
    'ROAD_GEOMETRY_DESC_ENCODED',
    'SPEED_ZONE',
    'SPEED_ZONE_CAT',
    'VEHICLE_YEAR',
    'PERSON_AGE_GROUP',
    'DAY_WEEK_DESC_ENCODED',
    'LIGHT_CONDITION_ENCODED',
    'SEVERITY_INDEX'
]

# 1. Descriptive statistics
print("\nDescriptive statistics for focus features:")
print(df[focus_features].describe(include='all'))

# 2. Distribution of Severity Index
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='SEVERITY_INDEX')
plt.title('Distribution of Accident Severity')
plt.xlabel('Severity Index')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('figures/severity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Speed Zone vs Severity
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='SPEED_ZONE', y='SEVERITY_INDEX')
plt.title('Speed Zone Categories vs Accident Severity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/speed_severity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Vehicle Year vs Severity
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='VEHICLE_YEAR', y='SEVERITY_INDEX', alpha=0.5)
plt.title('Vehicle Year vs Accident Severity')
plt.xlabel('Vehicle Year')
plt.ylabel('Severity Index')
plt.tight_layout()
plt.savefig('figures/vehicle_year_severity.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Correlation heatmap (numerical features)
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(15, 10))
correlation_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Numerical Features)')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Pair plot for selected features
pairplot_cols = ['SPEED_ZONE', 'VEHICLE_YEAR', 'SEVERITY_INDEX', 'DAY_WEEK_DESC_ENCODED']
g = sns.pairplot(df[pairplot_cols].dropna(), diag_kind='kde')
g.fig.suptitle('Pairwise Relationships Between Key Features', y=1.02)
plt.savefig('figures/pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Time of Day Analysis
if 'DAY_WEEK_DESC_ENCODED' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='DAY_WEEK_DESC_ENCODED', y='SEVERITY_INDEX')
    plt.title('Day of Week vs Accident Severity')
    plt.xlabel('Day of Week')
    plt.ylabel('Severity Index')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/day_severity_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Light Condition Analysis
if 'LIGHT_CONDITION_ENCODED' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='LIGHT_CONDITION_ENCODED', y='SEVERITY_INDEX')
    plt.title('Light Condition vs Accident Severity')
    plt.xlabel('Light Condition')
    plt.ylabel('Severity Index')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/light_severity_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. Mutual Information Scores Visualization
X = df[numeric_cols].drop(columns=['SEVERITY_INDEX'], errors='ignore').fillna(0)
y = df['SEVERITY_INDEX'].fillna(0).astype(int)
mi_scores = mutual_info_classif(X, y, discrete_features='auto')
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
mi_df = mi_df.sort_values('Mutual Information', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=mi_df, x='Mutual Information', y='Feature')
plt.title('Top 10 Features by Mutual Information Score')
plt.tight_layout()
plt.savefig('figures/mutual_information_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nMutual Information Scores:")
print(mi_df)

# 10. Correlation coefficients with visualization
correlations = []
for col in numeric_cols:
    if col != 'SEVERITY_INDEX':
        pearson_corr = df[[col, 'SEVERITY_INDEX']].corr(method='pearson').iloc[0,1]
        spearman_corr = df[[col, 'SEVERITY_INDEX']].corr(method='spearman').iloc[0,1]
        correlations.append({
            'Feature': col,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Pearson', ascending=False).head(10)

plt.figure(figsize=(12, 12))
x = np.arange(len(corr_df['Feature']))
width = 0.35

plt.bar(x - width/2, corr_df['Pearson'], width, label='Pearson')
plt.bar(x + width/2, corr_df['Spearman'], width, label='Spearman')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.title('Top 10 Features by Correlation with Severity')
plt.xticks(x, corr_df['Feature'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('figures/correlation_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nCorrelation coefficients with SEVERITY_INDEX:")
print(corr_df)

# 11. Random Forest Feature Importance
print("\nFitting Random Forest for feature importance visualization...")
# Prepare data (drop rows with missing values for simplicity)
rf_X = df[numeric_cols].drop(columns=['SEVERITY_INDEX'], errors='ignore').dropna()
rf_y = df.loc[rf_X.index, 'SEVERITY_INDEX'].astype(int)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(rf_X, rf_y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=rf_X.columns[indices])
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('figures/random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Random Forest feature importances plotted.")

# 12. PCA 2D Scatter Plot
print("\nPerforming PCA for 2D visualization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rf_X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=rf_y, palette='viridis', alpha=0.5)
plt.title('PCA 2D Scatter Plot Colored by Severity Index')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Severity Index')
plt.tight_layout()
plt.savefig('figures/pca_2d_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

print("PCA 2D scatter plot saved.")

print("\nAnalysis complete. All visualizations saved in the 'figures' directory.") 