## this is the older fast api script, for preservation reasons I made a copy of this
## into diabetes_analysis_mcp.py which is the newer 'state of the art' script
## that augments fastAPI with the MCP wrapper library.. just a few lines of code!

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Diabetes Dataset Analysis API",
    description="API for analyzing a diabetes dataset for model training",
    version="1.0.0",
    root_path="/apps-internal/680c05e23917d068b39dee86" 
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
try:
    df = pd.read_csv("diabetes_dataset.csv")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Dataset file 'diabetes_dataset.csv' not found. Please ensure the file exists in the correct location.")

# Models
class DatasetInfo(BaseModel):
    columns: List[str]
    num_rows: int
    num_features: int
    target_column: str
    target_distribution: Dict[str, int]

class FeatureStats(BaseModel):
    feature: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    missing_values: int
    
class CorrelationData(BaseModel):
    feature_pairs: List[Dict[str, float]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Dataset Analysis API"}

@app.get("/dataset/info", response_model=DatasetInfo)
def get_dataset_info():
    """Get basic information about the dataset"""
    target_dist = df['is_diabetic'].value_counts().to_dict()
    
    return {
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "num_features": len(df.columns) - 1,  # excluding target
        "target_column": "is_diabetic",
        "target_distribution": {str(k): int(v) for k, v in target_dist.items()}
    }

@app.get("/dataset/head")
def get_dataset_head(rows: int = Query(5, description="Number of rows to return")):
    """Get the first n rows of the dataset"""
    if rows <= 0:
        raise HTTPException(status_code=400, detail="Rows parameter must be positive")
    
    return df.head(rows).to_dict(orient="records")

@app.get("/dataset/describe")
def get_dataset_description():
    """Get basic statistics for all columns"""
    return json.loads(df.describe().to_json())

@app.get("/feature/stats", response_model=List[FeatureStats])
def get_feature_stats(features: Optional[List[str]] = Query(None)):
    """Get detailed statistics for specified features or all features"""
    if features is None:
        features = df.columns.tolist()
    else:
        for feature in features:
            if feature not in df.columns:
                raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found")
    
    stats = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            stats.append({
                "feature": feature,
                "mean": float(df[feature].mean()),
                "median": float(df[feature].median()),
                "std": float(df[feature].std()),
                "min": float(df[feature].min()),
                "max": float(df[feature].max()),
                "missing_values": int(df[feature].isna().sum())
            })
    
    return stats

@app.get("/correlation/matrix")
def get_correlation_matrix():
    """Get the correlation matrix for all numeric features"""
    return df.corr().to_dict()

@app.get("/correlation/target", response_model=Dict[str, float])
def get_target_correlation():
    """Get correlation coefficients between each feature and the target variable"""
    correlations = df.corr()['is_diabetic'].drop('is_diabetic').to_dict()
    return {k: float(v) for k, v in correlations.items()}

@app.get("/feature/histogram/{feature}")
def get_feature_histogram(feature: str, bins: int = Query(10, description="Number of bins")):
    """Get histogram data for a specific feature"""
    if feature not in df.columns:
        raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found")
    
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' is not numeric")
    
    hist, bin_edges = np.histogram(df[feature].dropna(), bins=bins)
    return {
        "counts": hist.tolist(),
        "bins": bin_edges.tolist(),
        "feature": feature
    }

@app.get("/feature/boxplot/{feature}")
def get_feature_boxplot(feature: str):
    """Get boxplot data for a specific feature"""
    if feature not in df.columns:
        raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found")
    
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' is not numeric")
    
    q1 = float(df[feature].quantile(0.25))
    q3 = float(df[feature].quantile(0.75))
    iqr = q3 - q1
    lower_bound = float(max(df[feature].min(), q1 - 1.5 * iqr))
    upper_bound = float(min(df[feature].max(), q3 + 1.5 * iqr))
    
    outliers = df[feature][(df[feature] < lower_bound) | (df[feature] > upper_bound)].tolist()
    
    return {
        "feature": feature,
        "min": float(df[feature].min()),
        "q1": q1,
        "median": float(df[feature].median()),
        "q3": q3,
        "max": float(df[feature].max()),
        "outliers": outliers[:100] if len(outliers) > 100 else outliers  # Limit number of outliers returned
    }

@app.get("/feature/comparison")
def get_feature_comparison(feature1: str, feature2: str, diabetic_only: bool = False):
    """Get data for comparing two features, optionally filtered by diabetic status"""
    if feature1 not in df.columns or feature2 not in df.columns:
        missing = []
        if feature1 not in df.columns:
            missing.append(feature1)
        if feature2 not in df.columns:
            missing.append(feature2)
        raise HTTPException(status_code=404, detail=f"Features not found: {', '.join(missing)}")
    
    if diabetic_only:
        filtered_df = df[df['is_diabetic'] == 1]
    else:
        filtered_df = df
    
    data = filtered_df[[feature1, feature2]].dropna().to_dict(orient="records")
    
    return {
        "feature1": feature1,
        "feature2": feature2,
        "diabetic_only": diabetic_only,
        "data": data[:1000]  # Limit data points returned to prevent large responses
    }

@app.get("/feature/group_analysis")
def get_group_analysis(feature: str, group_by: str = "is_diabetic"):
    """Group the data by a feature and calculate statistics for another feature in each group"""
    if feature not in df.columns or group_by not in df.columns:
        missing = []
        if feature not in df.columns:
            missing.append(feature)
        if group_by not in df.columns:
            missing.append(group_by)
        raise HTTPException(status_code=404, detail=f"Features not found: {', '.join(missing)}")
    
    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' must be numeric")
    
    result = {}
    for group, group_df in df.groupby(group_by):
        result[str(group)] = {
            "count": len(group_df),
            "mean": float(group_df[feature].mean()),
            "median": float(group_df[feature].median()),
            "std": float(group_df[feature].std()),
            "min": float(group_df[feature].min()),
            "max": float(group_df[feature].max())
        }
    
    return {
        "feature": feature,
        "grouped_by": group_by,
        "analysis": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)