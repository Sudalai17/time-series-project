# Troubleshooting Guide

## 🔍 Common Issues and Solutions

### Installation Issues

#### Issue 1: `pip install` fails
```
ERROR: Could not find a version that satisfies the requirement...
```

**Possible Causes**:
- Outdated pip version
- Python version too old
- Network connectivity issues

**Solutions**:
```bash
# Update pip first
python -m pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt

# If still fails, install packages one by one
pip install numpy pandas matplotlib seaborn scikit-learn shap
```

#### Issue 2: SHAP installation fails
```
ERROR: Failed building wheel for shap
```

**Solution**:
```bash
# Install build dependencies first
pip install --upgrade setuptools wheel

# On Windows, you may need Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install shap
pip install shap
```

#### Issue 3: Version conflicts
```
ERROR: package X requires Y>=Z.Z but you have Y==W.W
```

**Solution**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
fresh_env\Scripts\activate  # Windows
source fresh_env/bin/activate  # Mac/Linux

# Install requirements in clean environment
pip install -r requirements.txt
```

### Runtime Issues

#### Issue 4: ImportError
```
ModuleNotFoundError: No module named 'shap'
```

**Causes**:
- Package not installed
- Wrong Python environment
- Package name typo

**Solutions**:
```bash
# Check which Python you're using
which python  # Mac/Linux
where python  # Windows

# Verify packages installed
pip list | grep shap

# Install missing package
pip install shap

# If using virtual environment, make sure it's activated
```

#### Issue 5: Memory Error
```
MemoryError: Unable to allocate array
```

**Cause**: Insufficient RAM for SHAP computation

**Solutions**:

**Option 1**: Reduce SHAP sample size
```python
# In financial_forecasting.py, line ~445
# Change from:
sample_size = min(500, len(X_test))
# To:
sample_size = min(200, len(X_test))  # or even 100
```

**Option 2**: Reduce dataset size
```python
# In main(), line ~658
# Change from:
df = processor.generate_synthetic_data(n_samples=2000, n_features=50)
# To:
df = processor.generate_synthetic_data(n_samples=1000, n_features=30)
```

**Option 3**: Close other applications to free RAM

#### Issue 6: Script hangs/takes too long
```
Script has been running for 10+ minutes...
```

**Cause**: SHAP computation is computationally intensive

**Normal Timing**:
- 500 samples: ~60-90 seconds
- 1000 samples: ~120-180 seconds
- 2000 samples: ~240-300 seconds

**If taking longer**:
```python
# Reduce SHAP sample size (see Issue 5 solutions)
# Or reduce model complexity:

# In FinancialForecaster.train_model(), line ~360
self.model = GradientBoostingRegressor(
    n_estimators=100,  # Reduce from 200
    max_depth=3,       # Reduce from 5
    learning_rate=0.1,
    random_state=42
)
```

#### Issue 7: No plots displayed
```
Code runs but I don't see any plots
```

**Cause**: Plots are saved as files, not displayed

**Solution**: Check your directory for PNG files
```bash
ls -la *.png  # Mac/Linux
dir *.png     # Windows
```

**Expected files**:
- `global_feature_importance.png`
- `shap_summary_plot.png`
- `predictions_vs_actuals.png`

**To display plots interactively**:
```python
# Add this after plt.savefig() calls:
plt.show()

# Or run in Jupyter notebook
```

### Data Issues

#### Issue 8: "NaN values found"
```
ValueError: Input contains NaN
```

**Cause**: Feature engineering creates NaN values (lagging, rolling)

**Solution**: Already handled in code by `df.dropna()`, but if using your own data:
```python
# Check for NaN values
print(df.isnull().sum())

# Remove rows with NaN
df = df.dropna()

# Or fill NaN values
df = df.fillna(method='ffill')  # Forward fill
```

#### Issue 9: Wrong data format
```
KeyError: 'target_volatility'
```

**Cause**: Your CSV doesn't have required column

**Solution**: Rename your target column
```python
# If your target column is named differently
df = pd.read_csv('your_data.csv')
df = df.rename(columns={'your_target_col': 'target_volatility'})
```

#### Issue 10: Date/time index issues
```
TypeError: '<' not supported between instances of 'str' and 'int'
```

**Cause**: Date column treated as string

**Solution**:
```python
# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.sort_index()
```

### Output Issues

#### Issue 11: Empty or incomplete report
```
Report file is empty or missing sections
```

**Cause**: Script may have crashed before completing

**Solution**:
```bash
# Run script and save output
python financial_forecasting.py > output.log 2>&1

# Check log for errors
cat output.log  # Mac/Linux
type output.log  # Windows
```

#### Issue 12: Plots look bad/unreadable
```
Text is too small or overlapping
```

**Solution**: Adjust figure sizes
```python
# In create_visualizations(), around line ~500
plt.figure(figsize=(16, 10))  # Increase from (12, 8)

# Increase font sizes
plt.rcParams.update({'font.size': 12})

# Adjust layout
plt.tight_layout(pad=2.0)
```

#### Issue 13: CSV files have wrong encoding
```
UnicodeDecodeError when opening CSV
```

**Solution**:
```python
# Save with UTF-8 encoding explicitly
df.to_csv('output.csv', index=False, encoding='utf-8-sig')

# Read with proper encoding
df = pd.read_csv('file.csv', encoding='utf-8-sig')
```

### Performance Issues

#### Issue 14: Model performs poorly (R² < 0.5)
```
Test R² Score: 0.3456
```

**Possible Causes**:
- Not enough features
- Model too simple
- Data not suitable

**Solutions**:

**Increase model complexity**:
```python
self.model = GradientBoostingRegressor(
    n_estimators=300,     # Increase
    max_depth=7,          # Increase
    learning_rate=0.05,   # Decrease
    min_samples_split=10  # Decrease
)
```

**Add more features**:
```python
# In create_lagged_features()
lags = [1, 2, 3, 5, 10, 20]  # More lag periods

# In calculate_rolling_stats()
windows = [5, 10, 20, 50, 100]  # More window sizes
```

**Check data quality**:
```python
# Analyze correlations
print(df.corr()['target_volatility'].sort_values(ascending=False))

# Check for outliers
print(df.describe())
```

#### Issue 15: Model overfitting (train R² >> test R²)
```
Train R²: 0.95, Test R²: 0.55
```

**Solutions**:

**Reduce model complexity**:
```python
self.model = GradientBoostingRegressor(
    n_estimators=100,     # Decrease
    max_depth=3,          # Decrease
    learning_rate=0.1,
    min_samples_split=50  # Increase
)
```

**Add regularization**:
```python
self.model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,        # Use 80% of data per tree
    max_features='sqrt',  # Use subset of features
)
```

### SHAP Issues

#### Issue 16: SHAP values don't make sense
```
Top feature has zero importance but high SHAP value
```

**Cause**: May be interactions or non-linear effects

**Solution**:
```python
# Check feature interactions
shap.dependence_plot(
    feature_idx,
    shap_values,
    X_test_sample,
    interaction_index="auto"
)

# Verify SHAP values sum to prediction
expected_value = explainer.expected_value
prediction = expected_value + shap_values[i].sum()
```

#### Issue 17: SHAP plots are messy
```
Too many features shown, plot is cluttered
```

**Solution**:
```python
# In SHAP summary plot
shap.summary_plot(
    shap_values,
    X_test_sample,
    max_display=5,     # Show only top 5
    plot_size=(12, 6)  # Adjust size
)
```

### Platform-Specific Issues

#### Issue 18: Mac M1/M2 chip issues
```
Platform-specific errors on Apple Silicon
```

**Solution**:
```bash
# Use conda instead of pip
conda create -n finml python=3.9
conda activate finml
conda install numpy pandas scikit-learn matplotlib seaborn
pip install shap
```

#### Issue 19: Windows path issues
```
FileNotFoundError: [WinError 3] The system cannot find the path specified
```

**Solution**:
```python
# Use raw strings for Windows paths
import os

# Instead of:
path = "C:\Users\name\project"

# Use:
path = r"C:\Users\name\project"
# Or:
path = "C:/Users/name/project"
# Or:
path = os.path.join("C:", "Users", "name", "project")
```

#### Issue 20: Linux font issues
```
UserWarning: findfont: Font family ['...'] not found
```

**Solution**:
```bash
# Install font packages
sudo apt-get install fonts-liberation
sudo apt-get install msttcorefonts -qq

# Clear matplotlib cache
rm -rf ~/.cache/matplotlib
```

## 🔧 Debugging Tips

### Enable Verbose Output

```python
# Add at top of script
import logging
logging.basicConfig(level=logging.DEBUG)

# In model training
self.model = GradientBoostingRegressor(
    verbose=1,  # Show progress
    ...
)
```

### Check Intermediate Results

```python
# After each major step, add:
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(df.describe())
```

### Validate Data

```python
def validate_data(df, name="DataFrame"):
    print(f"\n=== Validating {name} ===")
    print(f"Shape: {df.shape}")
    print(f"NaN count: {df.isnull().sum().sum()}")
    print(f"Inf count: {np.isinf(df).sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    return df

# Use it:
df = validate_data(df, "After feature engineering")
```

### Profile Performance

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

# Use it:
@timer
def train_model(X, y):
    # training code
    pass
```

## 📞 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide** ✓
2. **Review error message carefully**
3. **Try suggested solutions**
4. **Search error message online**
5. **Check if issue is already reported**

### Information to Provide

When seeking help, include:
```
1. Operating System: [Windows 10 / macOS 13 / Ubuntu 22.04]
2. Python version: [python --version]
3. Package versions: [pip list | grep -E "(numpy|pandas|shap|sklearn)"]
4. Error message: [Full traceback]
5. What you tried: [List of solutions attempted]
6. Minimal code example: [Code that reproduces issue]
```

### Useful Commands

```bash
# Python version
python --version

# Package versions
pip list

# System info
python -m sysconfig

# Check script for syntax errors
python -m py_compile financial_forecasting.py

# Run with error details
python -u financial_forecasting.py 2>&1 | tee output.log
```

## 🎯 Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Import error | `pip install [package]` |
| Memory error | Reduce `sample_size` to 200 |
| Too slow | Reduce `n_estimators` to 100 |
| No plots | Check for PNG files in directory |
| Bad performance | Increase `n_estimators`, adjust `max_depth` |
| Overfitting | Add `subsample=0.8, max_features='sqrt'` |
| NaN errors | Check `df.dropna()` is called |
| SHAP errors | Reduce `sample_size` |
| File not found | Use absolute paths or check working directory |
| Version conflict | Create fresh virtual environment |

## ✅ Verification Checklist

After fixing issues, verify:
- [ ] Script runs without errors
- [ ] All 7 output files created
- [ ] Report has 200+ lines
- [ ] Plots are readable
- [ ] CSV files open correctly
- [ ] Performance metrics are reasonable (R² > 0.5)
- [ ] Execution time < 5 minutes

---

**Still having issues?** 
1. Re-read the README.md
2. Try the Jupyter notebook version
3. Start with smaller dataset
4. Check package versions match requirements.txt

**Good luck! 🚀**