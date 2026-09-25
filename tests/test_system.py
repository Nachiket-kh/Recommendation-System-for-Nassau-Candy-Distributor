from config import DATA_PATH
from src.data_loader import load_data, validate_schema
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.optimization import recommend_factory_and_shipping

def test_loader_and_schema():
    data = load_data(DATA_PATH)
    assert len(data) == 10194 and 'shipping_days' in data
    assert validate_schema(data)['missing_columns'] == []

def test_feature_engineering_and_cleaning():
    data = engineer_features(clean_data(load_data(DATA_PATH)))
    assert data['profit_margin'].notna().all()
    assert data['shipping_days'].min() >= 0

def test_missing_factory_response():
    result = recommend_factory_and_shipping({})
    assert not result['factory_available'] and 'requires' in result['message']
