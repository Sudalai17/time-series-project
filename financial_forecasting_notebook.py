# %% [markdown]
# # Interpretable Financial Time Series Forecasting
# 
# This notebook provides an interactive version of the financial forecasting project.
# You can run cells individually to explore each step of the pipeline.

# %% [markdown]
# ## Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("✓ All libraries imported successfully!")

# %% [markdown]
# ## Step 1: Data Generation
# 
# Generate synthetic high-dimensional financial dataset with 50+ features

# %%
def generate_financial_data(n_samples=2000, n_features=50):
    """Generate synthetic financial dataset."""
    print("Generating synthetic financial data...")
    
    time = np.arange(n_samples)
    base_signal = 0.5 * np.sin(2 * np.pi * time / 252) + 0.001 * time
    
    features = {}
    
    # Market indices
    for i in range(10):
        features[f'market_index_{i}'] = base_signal + np.random.randn(n_samples) * 0.1
    
    # Technical indicators
    for i in range(15):
        features[f'technical_{i}'] = np.random.randn(n_samples) * 0.5
    
    # Volatility measures
    for i in range(10):
        features[f'volatility_{i}'] = np.abs(np.random.randn(n_samples)) * 0.3
    
    # Macroeconomic indicators
    for i in range(15):
        features[f'macro_{i}'] = base_signal * 0.5 + np.random.randn(n_samples) * 0.2
    
    df = pd.DataFrame(features)
    
    # Create target variable
    target = (
        0.3 * df['market_index_0'] +
        0.2 * df['volatility_0'] ** 2 +
        0.15 * df['technical_5'] * df['macro_3'] +
        0.1 * np.sin(df['market_index_1'] * 5) +
        np.random.randn(n_samples) * 0.05
    )
    
    df['target_volatility'] = target
    
    print(f"✓ Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")
    return df

# Generate data
df = generate_financial_data()

# Display first few rows
print("\nFirst 5 rows:")
df.head()

# %% [markdown]
# ## Step 2: Feature Engineering

# %%
def create_lagged_features(df, columns, lags=[1, 5, 10]):
    """Create lagged features."""
    print(f"Creating lagged features...")
    for col in columns[:10]:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def calculate_rolling_stats(df, columns, windows=[5, 10, 20]):
    """Calculate rolling statistics."""
    print("Calculating rolling statistics...")
    for col in columns[:5]:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    return df

# Apply feature engineering
feature_cols = [col for col in df.columns if col != 'target_volatility']
df = create_lagged_features(df, feature_cols[:10], lags=[1, 5, 10])
df = calculate_rolling_stats(df, feature_cols[:5], windows=[5, 10, 20])

# Remove NaN values
df = df.dropna()

print(f"✓ After feature engineering: {df.shape[0]} samples, {df.shape[1]-1} features")

# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['target_volatility'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Target Volatility')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable')

plt.subplot(1, 2, 2)
plt.plot(df['target_volatility'].values[:500])
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Target Time Series (First 500 points)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 3: Data Preprocessing

# %%
# Separate features and target
X = df.drop(columns=['target_volatility'])
y = df['target_volatility']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Train-test split (preserve time series order)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

print(f"✓ Train set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")
print(f"✓ Number of features: {X_train.shape[1]}")

# %% [markdown]
# ## Step 4: Model Training

# %%
print("Training Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=20,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

print("✓ Model training completed!")

# %% [markdown]
# ## Step 5: Model Evaluation

# %%
# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
r2_score = model.score(X_test, y_test)

# Directional accuracy
train_direction = np.mean(np.sign(y_train_pred) == np.sign(y_train)) * 100
test_direction = np.mean(np.sign(y_test_pred) == np.sign(y_test)) * 100

# Display metrics
print("="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"\nTraining Set:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"  Directional Accuracy: {train_direction:.2f}%")

print(f"\nTest Set:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  Directional Accuracy: {test_direction:.2f}%")
print(f"  R² Score: {r2_score:.4f}")
print("="*60)

# Visualize predictions
plt.figure(figsize=(14, 6))

# Plot 1: Predictions vs Actuals
plt.subplot(1, 2, 1)
indices = range(min(200, len(y_test)))
plt.plot(indices, y_test.iloc[:len(indices)], label='Actual', alpha=0.7, linewidth=2)
plt.plot(indices, y_test_pred[:len(indices)], label='Predicted', alpha=0.7, linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Volatility')
plt.title('Predictions vs Actuals (First 200 points)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6: SHAP Analysis - Global Feature Importance

# %%
print("Computing SHAP values (this may take 1-2 minutes)...")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values on a sample
sample_size = min(500, len(X_test))
X_test_sample = X_test.iloc[:sample_size]
shap_values = explainer.shap_values(X_test_sample)

print(f"✓ SHAP values computed for {sample_size} samples")

# Calculate global feature importance
feature_importance = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(feature_importance)[-10:][::-1]

top_features_df = pd.DataFrame({
    'feature': [X_test.columns[i] for i in top_indices],
    'importance': [feature_importance[i] for i in top_indices],
    'mean_shap': [shap_values[:, i].mean() for i in top_indices]
})

print("\n" + "="*60)
print("TOP 10 MOST INFLUENTIAL FEATURES")
print("="*60)
print(top_features_df.to_string(index=False))
print("="*60)

# Visualize global importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_features_df)), top_features_df['importance'])
plt.yticks(range(len(top_features_df)), top_features_df['feature'])
plt.xlabel('Mean |SHAP Value|')
plt.title('Top 10 Most Influential Features - Global Importance')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: SHAP Summary Plot

# %%
# Create SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="dot", show=False, max_display=10)
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("• Features are sorted by importance (top to bottom)")
print("• Red = High feature value, Blue = Low feature value")
print("• Right side = Positive impact on prediction")
print("• Left side = Negative impact on prediction")

# %% [markdown]
# ## Step 8: Critical Time Point Analysis

# %%
# Identify critical time points with high prediction errors
errors = np.abs(y_test.values - y_test_pred)
critical_indices = np.argsort(errors)[-5:][::-1]

critical_points = pd.DataFrame({
    'index': [X_test.index[i] for i in critical_indices],
    'actual': [y_test.iloc[i] for i in critical_indices],
    'predicted': [y_test_pred[i] for i in critical_indices],
    'error': [errors[i] for i in critical_indices]
})

print("="*60)
print("CRITICAL TIME POINTS (Highest Prediction Errors)")
print("="*60)
print(critical_points.to_string(index=False))
print("="*60)

# Visualize critical points
plt.figure(figsize=(14, 6))
indices = range(min(200, len(y_test)))
plt.plot(indices, y_test.iloc[:len(indices)], label='Actual', alpha=0.7, linewidth=2)
plt.plot(indices, y_test_pred[:len(indices)], label='Predicted', alpha=0.7, linewidth=2)

# Mark critical points
critical_in_range = [i for i in critical_indices if i < len(indices)]
if critical_in_range:
    plt.scatter(critical_in_range, 
               [y_test.iloc[i] for i in critical_in_range],
               color='red', s=200, zorder=5, marker='X', 
               label='Critical Points', edgecolors='black', linewidths=2)

plt.xlabel('Time Step')
plt.ylabel('Volatility')
plt.title('Critical Prediction Points Highlighted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 9: Local Explanations for Critical Points

# %%
print("="*60)
print("LOCAL SHAP EXPLANATIONS FOR CRITICAL POINTS")
print("="*60)

local_analyses = []

for i, idx in enumerate(critical_indices[:5]):
    if idx >= len(X_test_sample):
        continue
    
    shap_values_point = shap_values[idx]
    top_contributions = np.argsort(np.abs(shap_values_point))[-5:][::-1]
    
    print(f"\nCritical Point #{i+1} (Index: {X_test.index[idx]}):")
    print(f"  Actual: {y_test.iloc[idx]:.4f}")
    print(f"  Predicted: {y_test_pred[idx]:.4f}")
    print(f"  Error: {errors[idx]:.4f}")
    print(f"  Top Contributing Features:")
    
    for rank, feat_idx in enumerate(top_contributions):
        feature_name = X_test.columns[feat_idx]
        shap_value = shap_values_point[feat_idx]
        feature_value = X_test_sample.iloc[idx, feat_idx]
        
        print(f"    {rank+1}. {feature_name}")
        print(f"       SHAP: {shap_value:+.4f} | Value: {feature_value:.4f}")
        
        local_analyses.append({
            'point': i+1,
            'feature': feature_name,
            'shap_value': shap_value,
            'feature_value': feature_value
        })

print("="*60)

# %% [markdown]
# ## Step 10: SHAP Force Plot for Individual Predictions

# %%
# Select an interesting critical point
if len(critical_indices) > 0:
    idx = critical_indices[0]
    if idx < len(X_test_sample):
        print(f"Force plot for critical point at index {X_test.index[idx]}")
        shap.initjs()
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X_test_sample.iloc[idx],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Step 11: Investment Strategy Insights

# %%
print("="*70)
print("ACTIONABLE INVESTMENT INSIGHTS")
print("="*70)

print("\n1. RISK MANAGEMENT:")
print("   • Monitor these top features continuously:")
for idx, row in top_features_df.head(5).iterrows():
    print(f"     - {row['feature']}")

print("\n2. POSITION SIZING STRATEGY:")
print(f"   Current volatility prediction: {y_test_pred[-1]:.4f}")
print(f"   Historical 75th percentile: {np.percentile(y_test, 75):.4f}")
if y_test_pred[-1] > np.percentile(y_test, 75):
    print("   → Recommendation: REDUCE position sizes by 30-50%")
else:
    print("   → Recommendation: Normal position sizing acceptable")

print("\n3. MODEL CONFIDENCE:")
shap_variance = np.var(np.abs(shap_values), axis=0).mean()
print(f"   SHAP variance (uncertainty): {shap_variance:.4f}")
if shap_variance < 0.05:
    print("   → High confidence - model explanations are consistent")
else:
    print("   → Moderate confidence - be cautious with predictions")

print("\n4. FEATURE MONITORING ALERTS:")
print("   Set alerts when these features exceed normal ranges:")
for idx, row in top_features_df.head(3).iterrows():
    feature_name = row['feature']
    feature_values = X_test[feature_name]
    p90 = np.percentile(feature_values, 90)
    print(f"   - {feature_name}: >|{p90:.2f}| (90th percentile)")

print("\n5. DIRECTIONAL TRADING:")
print(f"   Model directional accuracy: {test_direction:.2f}%")
if test_direction > 70:
    print("   → Strong signal - can use for directional bets")
else:
    print("   → Moderate signal - use for risk management only")

print("="*70)

# %% [markdown]
# ## Step 12: Export Results

# %%
# Save results
top_features_df.to_csv('top_features.csv', index=False)
critical_points.to_csv('critical_points.csv', index=False)
pd.DataFrame(local_analyses).to_csv('local_analyses.csv', index=False)

# Save model performance
performance_df = pd.DataFrame({
    'Metric': ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 
               'R² Score', 'Directional Accuracy'],
    'Value': [train_rmse, test_rmse, train_mae, test_mae, 
              r2_score, test_direction]
})
performance_df.to_csv('model_performance.csv', index=False)

print("✓ Results exported to CSV files:")
print("  - top_features.csv")
print("  - critical_points.csv")
print("  - local_analyses.csv")
print("  - model_performance.csv")

# %% [markdown]
# ## Summary and Conclusions

# %%
print("="*70)
print("PROJECT SUMMARY")
print("="*70)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   • R² Score: {r2_score:.4f} (explains {r2_score*100:.1f}% of variance)")
print(f"   • Test RMSE: {test_rmse:.4f}")
print(f"   • Directional Accuracy: {test_direction:.2f}%")

print(f"\n🎯 TOP 3 PREDICTIVE FEATURES:")
for idx, row in top_features_df.head(3).iterrows():
    print(f"   {idx+1}. {row['feature']} (importance: {row['importance']:.4f})")

print(f"\n⚠️  CRITICAL INSIGHTS:")
print(f"   • Identified {len(critical_points)} critical time points")
print(f"   • Average error at critical points: {critical_points['error'].mean():.4f}")
print(f"   • Model performs best on normal market conditions")

print(f"\n💼 INVESTMENT APPLICATION:")
print(f"   • Use for volatility-based position sizing")
print(f"   • Monitor top features for early warning")
print(f"   • Adjust hedging based on predicted volatility")
print(f"   • Review critical points for regime changes")

print("="*70)

# %%
print("\n✅ Analysis complete! All cells executed successfully.")
print("\nNext steps:")
print("1. Review the CSV files for detailed data")
print("2. Experiment with different model parameters")
print("3. Try using your own financial data")
print("4. Extend with additional interpretability methods")