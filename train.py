"""Train the Nassau Candy shipping recommendation model."""
from config import DATA_PATH, MODEL_DIR
from src.data_loader import load_data, get_dataset_summary
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.model_training import train_shipping_model
from src.eda import save_eda_plots
from config import ARTIFACT_DIR

if __name__ == "__main__":
    data = engineer_features(clean_data(load_data(DATA_PATH)))
    _, results = train_shipping_model(data, MODEL_DIR)
    save_eda_plots(data, ARTIFACT_DIR / "plots")
    summary = get_dataset_summary(data)
    print(f"Loaded {summary['rows']:,} rows; selected {results['selected_model']}.")
    for name, values in results["models"].items(): print(f"{name}: accuracy={values['accuracy']:.3f}, F1={values['f1_weighted']:.3f}, top-3={values['top_3_accuracy']:.3f}")
