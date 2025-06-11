# MTG Card Data Pipeline Documentation

## Pipeline Overview
This pipeline preprocesses the MTG Card dataset to prepare it for machine learning modeling. It handles categorical encoding, target encoding, outlier detection and treatment, feature scaling, and missing value imputation.

<img src="https://github.com/pudgypoultry/cis523Library/blob/ae2443e41de9f899719e6f25482cff40b7c70391/pipeline-image.png" width="70%" alt="Pipeline Diagram">

## Step-by-Step Design Choices

### 1. Banned or Not Mapping (`map_Banned_in_Commander`)
- **Transformer:** `CustomMappingTransformer('map_Banned_in_Commander', {'No': 0, 'Yes': 1})`
- **Design Choice:** Binary encoding of whether or not the card is banned in the most popular casual format of the game
- **Rationale:** Simple categorical mapping that preserves the binary nature of the feature without increasing dimensionality

### 2. Unique Power Mapping (`map_Power_is_Unique`)
- **Transformer:** `CustomMappingTransformer('PowerIsUnique', {'No': 0, 'Yes': 1})`
- **Design Choice:** Binary encoding of whether or not the card has a non-numerical, unique power stat
- **Rationale:** Simple categorical mapping that preserves the binary nature of the feature without increasing dimensionality

### 3. Unique Toughness Mapping (`map_Toughness_is_Unique`)
- **Transformer:** `CustomMappingTransformer('ToughnessIsUnique', {'No': 0, 'Yes': 1})`
- **Design Choice:** Binary encoding of whether or not the card has a non-numerical, unique toughness stat
- **Rationale:** 
  - Replaces the categorical 'Joined' feature with its relationship to the target variable
  - Smoothing=10 balances between using the global mean (high smoothing) and the category mean (low smoothing)
  - Helps address potential overfitting from rare categories

### 4. Target Encoding for Card Rarity (`target_rarity`)
- **Transformer:** `CustomTargetTransformer(col='Printed Rarity', smoothing=10)`
- **Design Choice:** Target encoding with smoothing factor of 10
- **Rationale:** 
  - Replaces the categorical 'Printed Rarity' feature with its relationship to the target variable
  - Smoothing=10 balances between using the global mean (high smoothing) and the category mean (low smoothing)
  - Helps address potential overfitting from rare categories

### 5. Outlier Treatment for Card Power (`tukey_Power`)
- **Transformer:** `CustomTukeyTransformer(target_column='Power', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:**
  - Power statistic has high variability and legitimate outliers with grouping closer to 0
  - Outer fence preserves most of the original distribution while handling extreme values

### 6. Outlier Treatment for Card Toughness (`tukey_Toughness`)
- **Transformer:** `CustomTukeyTransformer(target_column='Toughness', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:**
  - Toughness statistic has high variability and legitimate outliers with grouping closer to 1
  - Outer fence preserves most of the original distribution while handling extreme values

### 6. Power Scaling (`scale_Power`)
- **Transformer:** `CustomRobustTransformer(target_column='Age')`
- **Design Choice:** Robust scaling for Age feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for Power which may not follow normal distribution

### 7. Toughness Scaling (`scale_Toughness`)
- **Transformer:** `CustomRobustTransformer(target_column='Fare')`
- **Design Choice:** Robust scaling for Fare feature
- **Rationale:**
  - Toughness has high variability and skewed distribution
  - Robust scaling reduces influence of remaining outliers after Tukey treatment

### 8. Imputation (`impute`)
- **Transformer:** `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice:** KNN imputation with 5 neighbors
- **Rationale:**
  - Uses relationships between features to estimate missing values
  - k=5 balances between too few neighbors (overfitting) and too many (underfitting)
  - Since no columns are missing values, this shouldn't actually effect much, but leaving it in for completeness

## Pipeline Execution Order Rationale
1. Categorical encoding first to prepare for subsequent numerical operations
2. Target encoding next as it requires original categorical values
3. Outlier treatment before scaling to prevent outliers from affecting scaling parameters
4. Scaling before imputation so that distance metrics in KNN aren't skewed by unscaled features
5. Imputation last to fill missing values using all preprocessed features

## Performance Considerations
- RobustScaler instead of StandardScaler due to presence of outliers
- KNN imputation instead of simple imputation to preserve relationships between features
- Target encoding with smoothing for categorical features with many levels
