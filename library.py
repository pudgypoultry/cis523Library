from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import category_encoders as ce
import types
import warnings
import joblib
import sklearn
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from feature_engine.encoding import MeanEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.preprocessing import _encoders
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import ParameterGrid

sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result : pd.DataFrame = self.transform(X)
        return result


class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that renames columns in a pandas DataFrame.
    """

    def __init__(self, mapping_dict: Dict[str, str]) -> None:
        """
        Initializes the CustomRenamingTransformer.

        Args:
            renaming_dict: A dictionary where keys are the old column names and values are the new column names.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict = mapping_dict

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> CustomRenamingTransformer:
        """
        Fits the transformer. This method does nothing as it is not needed for renaming columns.

        Args:
            X: The input DataFrame.
            y: Ignored.

        Returns:
            self: The transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by renaming the columns according to the mapping_dict.

        Args:
            X: The input DataFrame.

        Returns:
            A new DataFrame with the renamed columns.

        Raises:
            AssertionError: If any of the columns specified in the mapping_dict do not exist in the input DataFrame.
        """

        # Check if all keys in mapping_dict are present in DataFrame columns
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        # assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

        missing_cols = set(self.mapping_dict.keys()) - set(X.columns)
        assert not missing_cols, f"Columns not found in DataFrame: {missing_cols}"

        X_ = X.copy()
        X_ = X_.rename(columns=self.mapping_dict)
        return X_


class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        for item in column_list:
            assert isinstance(item, str), f'CustomDropColumnsTransformer.transform unknown columns to keep: {item}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> CustomDropColumnsTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        #Warning: DropColumnsTransformer does not contain these columns to drop: {'Rating'}.
        # based on the self.action, drop or keep all items in self.column_list
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

        X_1 = X.copy()

        if self.action == 'drop':
          extras = []
          for item in self.column_list:
            if item not in X.columns:
              print(f'Warning: CustomDropColumnsTransformer.transform unknown columns to keep: {item}')
              extras.append(item)
          if len(extras) > 0:
            for item in extras:
              self.column_list.remove(item)
          X_1 = X_1.drop(columns=self.column_list)
        elif self.action == 'keep':
          for item in self.column_list:
            assert item in X.columns, f'CustomDropColumnsTransformer.transform unknown columns to keep: {item}'
          X_1 = X_1[self.column_list]

        # print(X_2.head())
        return X_1

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        result = self.transform(X)
        return result



class CustomOHETransformer(BaseEstimator, TransformerMixin):

    def __init__(self, target_column: str) -> None:
        assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected dictionary but got {type(target_column)} instead.'
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> CustomRenamingTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Check if all keys in mapping_dict are present in DataFrame columns
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?

        X_1 = X.copy()

        X_2 = pd.get_dummies(X_1,
                            prefix=self.target_column,
                            prefix_sep='_',
                            columns=[self.target_column],
                            dummy_na=False,
                            drop_first=False,
                            dtype=int
                            )
        # print(X_2.head())
        return X_2

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        result = self.transform(X)
        return result


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, column: str):
        self.has_been_fit = False
        self.target_column = column
        self.high_wall = None
        self.low_wall = None
        self.mean = None
        self.std = None

    def fit(self, X: pd.DataFrame, y=None):

        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        #assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected int or float in column {self.target_column}'
        X2 = X.copy()
        self.has_been_fit = True

        # Compute the mean and standard deviation of the column
        self.mean = X2[self.target_column].mean()
        self.std = X2[self.target_column].std()

        # Compute the low and high boundaries
        self.low_wall = self.mean - 3 * self.std
        self.high_wall = self.mean + 3 * self.std


    def transform(self, X: pd.DataFrame):

        assert self.has_been_fit, 'Sigma3Transformer.fit has not been called.'
        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        #assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected int or float in column {self.target_column}'

        X2 = X.copy()
        X2[self.target_column] = X2[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        #X = X.reset_index(drop=True)
        return X2


    def fit_transform(self, X: pd.DataFrame, y=None):
        X2 = X.copy()
        self.fit(X2, y)
        return self.transform(X2)


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
#     """
#     A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.
#     This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
#     It clips values in the target column based on Tukey's inner or outer fences.
#     Parameters
#     ----------
#     target_column : Hashable
#         The name of the column to apply Tukey's fences on.
#     fence : Literal['inner', 'outer'], default='outer'
#         Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).
#     Attributes
#     ----------
#     inner_low : Optional[float]
#         The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
#     outer_low : Optional[float]
#         The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
#     inner_high : Optional[float]
#         The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
#     outer_high : Optional[float]
#         The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).
#     Examples
#     --------
#     >>> import pandas as pd
#     >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
#     >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
#     >>> transformed_df = tukey_transformer.fit_transform(df)
#     >>> transformed_df
#     """

    def __init__(self, target_column: str, fence: str = 'outer'):
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.outer_low = None
        self.inner_high = None
        self.outer_high = None
        self.has_been_fit = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Compute the inner and outer fences for Tukey's method.
        """
        # assert isinstance(X, pd.DataFrame), f"Expected DataFrame, got {type(X)}"
        assert self.target_column in X.columns, f"Column '{self.target_column}' not found"
        assert self.fence in ['inner', 'outer'], f"Invalid 'fence' value. Must be 'inner' or 'outer'."
        #assert pd.api.types.is_numeric_dtype(X[self.target_column]), f"Column '{self.target_column}' must be numeric"

        X2 = X.copy()
        self.has_been_fit = True

        q1 = X2[self.target_column].quantile(0.25)
        q3 = X2[self.target_column].quantile(0.75)
        the_iqr = q3 - q1

        self.inner_low = q1 - 1.5 * the_iqr
        self.outer_low = q1 - 3 * the_iqr
        self.inner_high = q3 + 1.5 * the_iqr
        self.outer_high = q3 + 3 * the_iqr
        #return self

    def transform(self, X: pd.DataFrame):
        """
        Clip values in the target column based on the computed fences.
        """
        assert self.has_been_fit, "Fit method has not been called."
        # assert isinstance(X, pd.DataFrame), f"Expected DataFrame, got {type(X)}"
        assert self.target_column in X.columns, f"Column '{self.target_column}' not found"
        #assert pd.api.types.is_numeric_dtype(X[self.target_column]), f"Column '{self.target_column}' must be numeric"
        X2 = X.copy()
        low_bound = -float('inf')
        high_bound = float('inf')
        if self.fence == 'inner':
            low_bound = self.inner_low
            high_bound = self.inner_high
        elif self.fence == 'outer':
            low_bound = self.outer_low
            high_bound = self.outer_high
        else:
            raise ValueError("Invalid 'fence' value. Must be 'inner' or 'outer'.")
        #print("Clipping with respect to: ", self.fence)
        #print("Low bound: ", low_bound)
        #print("High bound: ", high_bound)
        X2[self.target_column] = X2[self.target_column].clip(lower=low_bound, upper=high_bound)
        X2 = X2.reset_index(drop=True)

        return X2

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Compute the inner and outer fences and clip values in the target column.
        """
        X2 = X.copy()
        self.fit(X2, y)
        return self.transform(X2)


class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    """Applies robust scaling to a specified column in a pandas DataFrame.
      This transformer calculates the interquartile range (IQR) and median
      during the `fit` method and then uses these values to scale the
      target column in the `transform` method.

      Parameters
      ----------
      column : str
          The name of the column to be scaled.

      Attributes
      ----------
      target_column : str
          The name of the column to be scaled.
      iqr : float
          The interquartile range of the target column.
      med : float
          The median of the target column.
    """
    def __init__(self, target_column):
        self.target_column = target_column
        self.iqr = None
        self.med = None
        self.has_fit = False

    def fit(self, X, y=None):
        assert self.target_column in X.columns, f"Column '{self.target_column}' not found in DataFrame."

        self.has_fit = True
        # Calculate the IQR and median for the specified column, handling potential errors
        try:
            self.iqr = X[self.target_column].quantile(0.75) - X[self.target_column].quantile(0.25)
            self.med = X[self.target_column].median()
            # Check if iqr is zero
            if self.iqr == 0:
              print(f"Warning: IQR for column '{self.target_column}' is zero. Scaling will be skipped.")
              
        except Exception as e:
          print(f"Error during fit: {e}")
          self.iqr = None
          self.med = None
          
        return self


    def transform(self, X):
        # Create a copy to avoid modifying the original DataFrame
        X_transformed = X.copy()

        # Check if the specified column exists
        assert self.has_fit, "NotFittedError: This CustomRobustTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        assert self.target_column in X_transformed.columns, f"Column '{self.target_column}' not found in DataFrame."
            
        if self.iqr is not None and self.iqr != 0 :
            # Apply robust scaling to the specified column
            X_transformed[self.target_column] = (X_transformed[self.target_column] - self.med) / self.iqr
        
        return X_transformed
    
    def fit_transform(self, X, y=None, **fit_params):
       return super().fit_transform(X, y, **fit_params)


class CustomKNNTransformer(BaseEstimator, TransformerMixin):
    """Imputes missing values using KNN.

    This transformer wraps the KNNImputer from scikit-learn and hard-codes
    add_indicator to be False. It also ensures that the input and output
    are pandas DataFrames.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. Possible values:
        "uniform" : uniform weights. All points in each neighborhood
        are weighted equally.
        "distance" : weight points by the inverse of their distance.
        in this case, closer neighbors of a query point will have a
        greater influence than neighbors which are further away.
    """
    #your code below
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, add_indicator=False)
        self.has_been_fit = False

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "Input must be a dataframe"
        if self.n_neighbors > len(X):
            print("Warning: K must be greater than number of samples")
        self.has_been_fit = True
        self.knn_imputer.fit(X, y)
        return self

    def transform(self, X, y=None):
        assert self.has_been_fit, "NotFittedError: This CustomKNNTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        assert isinstance(X, pd.DataFrame), "Input must be a dataframe"
        X = pd.DataFrame(self.knn_imputer.transform(X), columns=X.columns)
        return X

    def fit_transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "Input must be a dataframe"
        return self.fit(X).transform(X)


class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float =10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        #Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col+'_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()    #dictionary of unique values in the column col and their counts
        means = X_[target].groupby(X_[self.col]).mean().to_dict() #dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)


def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200
                  ) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df, labels, test_size=0.2, shuffle=True,
            random_state=i, stratify=labels  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(np.array(Var) - mean_f1_ratio).argmin()  # Index of value closest to mean

    return rs_value, Var


def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below
  table_after_y_drop = original_table.drop(columns=label_column_name)
  labels = original_table[label_column_name].to_list()

  X_train, X_test, y_train, y_test = train_test_split(table_after_y_drop, 
                                                      labels, 
                                                      test_size=ts, 
                                                      shuffle=True,
                                                      random_state=rs, 
                                                      stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train, y_train)
  X_test_transformed = the_transformer.transform(X_test)

  X_train_numpy = X_train_transformed.to_numpy()
  X_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)
  
  return X_train_numpy, X_test_numpy, y_train_numpy,  y_test_numpy


titanic_variance_based_split = 107
customer_variance_based_split = 113 

url = 'https://raw.githubusercontent.com/fickas/asynch_models/refs/heads/main/datasets/titanic_trimmed.csv'
titanic_trimmed = pd.read_csv(url)
titanic_features = titanic_trimmed.drop(columns='Survived')
titanic_labels = titanic_trimmed['Survived'].to_list()

titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', CustomTargetTransformer(col='Joined', smoothing=10)),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer(target_column='Age')),
    ('scale_fare', CustomRobustTransformer(target_column='Fare')),
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)


customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', CustomTargetTransformer(col='ISP')),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer(target_column='Age')), #from 5
    ('scale_time spent', CustomRobustTransformer(target_column='Time Spent')), #from 5
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)


card_transformer = Pipeline(steps=[
    ('map_Power_is_Unique', CustomMappingTransformer('PowerIsUnique', {'No': 0, 'Yes': 1})),
    ('map_Toughness_is_Unique', CustomMappingTransformer('ToughnessIsUnique', {'No': 0, 'Yes': 1})),
    ('target_rarity', CustomTargetTransformer(col='Printed Rarity', smoothing=10)),
    ('tukey_Power', CustomTukeyTransformer(target_column='Power', fence='outer')),
    ('tukey_Toughness', CustomTukeyTransformer(target_column='Toughness', fence='outer')),
    ('scale_Power', CustomRobustTransformer(target_column='Power')),
    ('scale_Toughness', CustomRobustTransformer(target_column='Toughness')),
    ('impute', CustomKNNTransformer(n_neighbors=5)),
    ], verbose=True)

X_train, X_test, y_train, y_test = train_test_split(titanic_features,   #all columns except target column
                                                    titanic_labels,             #a list of values from target column, e.g., Survived
                                                    test_size=0.2,      #percent to put in test set
                                                    shuffle=True,        #whether to randomly shuffle rows (and labels) first
                                                    random_state=0)     #a seed for the random shuffle

fitted_pipeline = titanic_transformer.fit(X_train, y_train)  #notice just fit method called
joblib.dump(fitted_pipeline, 'fitted_pipeline.pkl')  #and next move to GitHub

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs, ts=.2)


def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
    return dataset_setup(customer_table, 'Rating', transformer, rs, ts=.2)


def threshold_results(thresh_list, actuals, predicted):
    result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'auc', 'accuracy'])
    for t in thresh_list:
        yhat = [1 if v >=t else 0 for v in predicted]
        precision = precision_score(actuals, yhat, zero_division=0)
        recall = recall_score(actuals, yhat, zero_division=0)
        f1 = f1_score(actuals, yhat)
        accuracy = accuracy_score(actuals, yhat)
        auc = roc_auc_score(actuals, predicted)
        result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'auc': auc, 'accuracy':accuracy}
    
    result_df = result_df.round(2)

    headers = {
      "selector": "th:not(.index_name)",
      "props": "background-color: #800000; color: white; text-align: center"
    }
    properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

    fancy_df = result_df.style.highlight_max(color = 'pink', axis = 0).format(precision=2).set_properties(**properties).set_table_styles([headers])
    return (result_df, fancy_df)


def halving_search(model, grid, x_train, y_train, factor=3, min_resources="exhaust", scoring='roc_auc'):
    #your code below
    halving_cv = HalvingGridSearchCV(
        model, grid,
        scoring=scoring,
        n_jobs=-1,
        min_resources=min_resources,
        factor=factor,
        cv=5, random_state=1234,
        refit=True
        )

    halving_cv.fit(x_train, y_train)

    df = pd.DataFrame(halving_cv.cv_results_)

    return halving_cv


def sort_grid(grid):
  sorted_grid = grid.copy()

  #sort values - note that this will expand range for you
  for k,v in sorted_grid.items():
    sorted_grid[k] = sorted(sorted_grid[k], key=lambda x: (x is None, x))  #handles cases where None is an alternative value

  #sort keys
  sorted_grid = dict(sorted(sorted_grid.items()))

  return sorted_grid
