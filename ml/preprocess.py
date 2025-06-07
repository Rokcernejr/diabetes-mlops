import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

# Define categorical columns based on the diabetes dataset
CATEGORICAL_COLS = [
    'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
]

NUMERICAL_COLS = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

# Diagnosis columns (require special processing)
DIAGNOSIS_COLS = ['diag_1', 'diag_2', 'diag_3']

def load_diabetes_data(file_path: Path) -> pd.DataFrame:
    """Load and perform initial validation of diabetes dataset"""
    logger.info(f"Loading data from {file_path}")
    
    # Load data with comprehensive missing value handling
    df = pd.read_csv(file_path, na_values=['?', 'Unknown', 'NULL', ''])
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Log missing values summary
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Missing values found in {missing_counts[missing_counts > 0].shape[0]} columns")
        for col, count in missing_counts[missing_counts > 0].items():
            logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the target variable with validation"""
    df = df.copy()
    
    if 'readmitted' not in df.columns:
        raise ValueError("Target column 'readmitted' not found in dataset")
    
    # Log original target distribution
    original_counts = df['readmitted'].value_counts()
    logger.info(f"Original target distribution: {original_counts.to_dict()}")
    
    # Convert readmitted to binary (< 30 days = 1, else = 0)
    readmit_mapping = {
        'NO': 0,
        '<30': 1,
        '>30': 0
    }
    
    df['readmitted'] = df['readmitted'].map(readmit_mapping)
    
    # Check for unmapped values
    unmapped = df['readmitted'].isnull().sum()
    if unmapped > 0:
        logger.warning(f"Found {unmapped} unmapped target values, dropping these rows")
        df = df.dropna(subset=['readmitted'])
    
    # Log final target distribution
    final_counts = df['readmitted'].value_counts()
    logger.info(f"Final target distribution: {final_counts.to_dict()}")
    
    # Check class balance
    minority_class_pct = min(final_counts) / final_counts.sum() * 100
    if minority_class_pct < 5:
        logger.warning(f"Severe class imbalance: minority class is {minority_class_pct:.1f}%")
    
    return df

def process_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Process diagnosis codes into meaningful medical categories"""
    df = df.copy()
    
    def categorize_diagnosis(code):
        """Categorize ICD-9 diagnosis codes into broad medical categories"""
        if pd.isna(code) or code in ['Unknown', '']:
            return 'Unknown'
        
        # Convert to string and extract first 3 characters for ICD-9 classification
        code_str = str(code)[:3]
        
        try:
            code_num = float(code_str)
            
            # ICD-9 code ranges for major disease categories
            if 250 <= code_num <= 259:
                return 'Diabetes'
            elif 390 <= code_num <= 459:
                return 'Circulatory'
            elif 460 <= code_num <= 519:
                return 'Respiratory'
            elif 520 <= code_num <= 579:
                return 'Digestive'
            elif 580 <= code_num <= 629:
                return 'Genitourinary'
            elif 140 <= code_num <= 239:
                return 'Neoplasms'
            elif 710 <= code_num <= 739:
                return 'Musculoskeletal'
            elif 290 <= code_num <= 319:
                return 'Mental'
            elif 320 <= code_num <= 389:
                return 'Nervous'
            elif 680 <= code_num <= 709:
                return 'Skin'
            else:
                return 'Other'
        except (ValueError, TypeError):
            return 'Other'
    
    # Process each diagnosis column
    diagnosis_categories_created = []
    for diag_col in DIAGNOSIS_COLS:
        if diag_col in df.columns:
            new_col = f'{diag_col}_category'
            df[new_col] = df[diag_col].apply(categorize_diagnosis)
            diagnosis_categories_created.append(new_col)
    
    # Create diabetes-specific indicators
    for diag_col in DIAGNOSIS_COLS:
        if diag_col in df.columns:
            df[f'{diag_col}_is_diabetes'] = (df[f'{diag_col}_category'] == 'Diabetes').astype(int)
    
    # Drop original diagnosis columns (too many unique values for ML)
    df = df.drop(columns=[col for col in DIAGNOSIS_COLS if col in df.columns])
    
    logger.info(f"Processed diagnosis codes: created {len(diagnosis_categories_created)} category columns")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive missing value handling with column-specific strategies"""
    df = df.copy()
    
    # Drop columns with too many missing values (>50%)
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    
    # Never drop the target column
    if 'readmitted' in cols_to_drop:
        cols_to_drop.remove('readmitted')
    
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >50% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Fill missing values with column-specific strategies
    for col in df.columns:
        if col == 'readmitted':
            continue
            
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
            
        if col in NUMERICAL_COLS and col in df.columns:
            # Fill numerical with median
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logger.info(f"Filled {missing_count} missing values in {col} with median: {median_value}")
            
        elif col in CATEGORICAL_COLS and col in df.columns:
            # Fill categorical with mode or 'Unknown'
            mode_values = df[col].mode()
            fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Filled {missing_count} missing values in {col} with: {fill_value}")
            
        else:
            # Default strategy for other columns
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    logger.info(f"After missing value handling: {df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced engineered features"""
    df = df.copy()
    initial_cols = len(df.columns)
    
    # Medication-related features
    if 'change' in df.columns:
        df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    
    if 'num_medications' in df.columns:
        df['high_medication_count'] = (df['num_medications'] > df['num_medications'].median()).astype(int)
        df['very_high_medications'] = (df['num_medications'] > df['num_medications'].quantile(0.9)).astype(int)
    
    # Hospital utilization features
    if 'number_emergency' in df.columns:
        df['has_emergency_visits'] = (df['number_emergency'] > 0).astype(int)
        df['frequent_emergency'] = (df['number_emergency'] > 1).astype(int)
    
    if 'time_in_hospital' in df.columns:
        df['long_hospital_stay'] = (df['time_in_hospital'] > df['time_in_hospital'].median()).astype(int)
        df['very_long_stay'] = (df['time_in_hospital'] > df['time_in_hospital'].quantile(0.75)).astype(int)
    
    # Healthcare complexity features
    if 'number_diagnoses' in df.columns:
        df['many_diagnoses'] = (df['number_diagnoses'] > df['number_diagnoses'].median()).astype(int)
    
    if 'num_procedures' in df.columns:
        df['had_procedures'] = (df['num_procedures'] > 0).astype(int)
    
    # Age-related features (if age is categorical like "[60-70)")
    if 'age' in df.columns:
        df['is_elderly'] = df['age'].astype(str).str.contains('70|80|90', na=False).astype(int)
        df['is_middle_aged'] = df['age'].astype(str).str.contains('50|60', na=False).astype(int)
        df['is_young'] = df['age'].astype(str).str.contains('10|20|30', na=False).astype(int)
    
    # Diabetes management features
    if 'diabetesMed' in df.columns:
        df['on_diabetes_medication'] = (df['diabetesMed'] == 'Yes').astype(int)
    
    # Lab test features
    if 'A1Cresult' in df.columns:
        df['a1c_tested'] = (~df['A1Cresult'].isin(['None', 'Unknown'])).astype(int)
        df['a1c_high'] = (df['A1Cresult'] == '>7').astype(int)
    
    # Create interaction features
    if all(col in df.columns for col in ['num_medications', 'time_in_hospital']):
        df['meds_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1)  # +1 to avoid division by zero
    
    new_features = len(df.columns) - initial_cols
    logger.info(f"Created {new_features} engineered features")
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features with proper handling"""
    df = df.copy()
    
    # Get categorical columns that exist in the dataframe
    categorical_cols = [col for col in df.columns 
                       if df[col].dtype == 'object' and col != 'readmitted']
    
    if not categorical_cols:
        logger.info("No categorical columns to encode")
        return df
    
    # Log categorical column information
    for col in categorical_cols:
        unique_count = df[col].nunique()
        logger.info(f"Encoding {col}: {unique_count} unique values")
        
        # Warn about high cardinality columns
        if unique_count > 50:
            logger.warning(f"{col} has {unique_count} unique values - consider further preprocessing")
    
    # One-hot encode categorical columns
    logger.info(f"One-hot encoding {len(categorical_cols)} categorical columns")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)
    
    logger.info(f"After encoding: {df.shape}")
    return df

def final_data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Final validation and cleanup of the processed dataset"""
    df = df.copy()
    
    # Ensure target column exists
    if 'readmitted' not in df.columns:
        raise ValueError("Target column 'readmitted' missing after preprocessing")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col != 'readmitted']
    
    # Ensure all features are numeric
    for col in feature_cols:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'object':
            logger.warning(f"Non-numeric column after encoding: {col}")
    
    # Check for any remaining missing values
    missing_final = df.isnull().sum().sum()
    if missing_final > 0:
        logger.warning(f"Found {missing_final} remaining missing values")
        # Drop rows with any missing values as final cleanup
        before_drop = len(df)
        df = df.dropna()
        after_drop = len(df)
        logger.info(f"Dropped {before_drop - after_drop} rows with missing values")
    
    # Validate target variable
    target_values = df['readmitted'].unique()
    if not all(val in [0, 1] for val in target_values):
        raise ValueError(f"Invalid target values: {target_values}")
    
    # Check for constant columns (no variance)
    constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]
    if constant_cols:
        logger.warning(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
        feature_cols = [col for col in feature_cols if col not in constant_cols]
    
    logger.info(f"Final validation complete: {df.shape}")
    return df

def preprocess_diabetes_data(file_path: Path, output_path: Path = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete preprocessing pipeline for diabetes readmission data
    
    Returns:
        df: Processed DataFrame ready for ML
        feature_cols: List of feature column names
    """
    logger.info("=" * 50)
    logger.info("Starting comprehensive diabetes data preprocessing pipeline")
    logger.info("=" * 50)
    
    # Load data
    df = load_diabetes_data(file_path)
    
    # Remove unnecessary identifier columns
    id_cols_to_remove = ['encounter_id', 'patient_nbr']
    existing_id_cols = [col for col in id_cols_to_remove if col in df.columns]
    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)
        logger.info(f"Removed identifier columns: {existing_id_cols}")
    
    # Preprocess target variable
    df = preprocess_target(df)
    
    # Process diagnosis codes
    df = process_diagnosis_codes(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering (before encoding to preserve original columns)
    df = feature_engineering(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Final validation and cleanup
    df = final_data_validation(df)
    
    # Get final feature list
    feature_cols = [col for col in df.columns if col != 'readmitted']
    
    # Log final statistics
    logger.info("=" * 50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Target distribution: {df['readmitted'].value_counts().to_dict()}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Save processed data if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    
    logger.info("Preprocessing pipeline completed successfully")
    return df, feature_cols








#Generalized Verions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

# Define categorical columns
CATEGORICAL_COLS = [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'max_glu_serum', 'A1Cresult', 'metformin',
    'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed'
]

NUMERICAL_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

def load_diabetes_data(file_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path, na_values=['?', 'Unknown'])
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df

def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Convert readmitted to binary (< 30 days = 1, else = 0)
    readmit_mapping = {
        'NO': 0,
        '<30': 1,
        '>30': 0
    }
    
    df['readmitted'] = df['readmitted'].map(readmit_mapping)
    logger.info(f"Target distribution: {df['readmitted'].value_counts().to_dict()}")
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Drop columns with too many missing values (>50%)
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping columns with >50% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Fill missing values
    for col in df.columns:
        if col in NUMERICAL_COLS:
            df[col] = df[col].fillna(df[col].median())
        elif col in CATEGORICAL_COLS:
            df[col] = df[col].fillna('Unknown')
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # One-hot encode categorical columns
    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    logger.info(f"After encoding: {len(df.columns)} columns")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Medication change indicator
    if 'change' in df.columns:
        df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    
    # High medication count
    if 'num_medications' in df.columns:
        df['high_medication_count'] = (df['num_medications'] > 15).astype(int)
    
    # Emergency visits indicator
    if 'number_emergency' in df.columns:
        df['has_emergency_visits'] = (df['number_emergency'] > 0).astype(int)
    
    # Long hospital stay
    if 'time_in_hospital' in df.columns:
        df['long_hospital_stay'] = (df['time_in_hospital'] > 7).astype(int)
    
    logger.info("Created engineered features")
    return df

def preprocess_diabetes_data(file_path: Path, output_path: Path = None) -> Tuple[pd.DataFrame, List[str]]:
    # Load data
    df = load_diabetes_data(file_path)
    
    # Preprocess target
    df = preprocess_target(df)
    
    # Remove unnecessary columns
    cols_to_remove = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != 'readmitted']
    
    # Save processed data
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    
    logger.info(f"Preprocessing complete: {len(df)} rows, {len(feature_cols)} features")
    return df, feature_cols
"""