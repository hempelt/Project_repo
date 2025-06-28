import gradio as gr
import pandas as pd
import mlflow
import mlflow.sklearn

# Set the MLflow experiment name
mlflow.set_experiment("tm_prediction_experiment")

# Load the latest run ID from a file
with open("latest_run.txt", "r") as f:
    run_id = f.read().strip()

# Load the trained model from MLflow using the run ID
model_uri = f"runs:/{run_id}/gradient_boosting_model"
model = mlflow.sklearn.load_model(model_uri)

# Load validation results (cross-validation metrics)
cv_results = pd.read_csv(
    r'C:/Users/hempe/Studium/Real_Project/Project_repo/models/validation_results.csv'
)

# Format the cross-validation results for display
validation_text = (
    f"### 🔍 Model Validation (5-Fold CV)\n"
    f"- **RMSE**: {cv_results['RMSE'].mean():.2f} ± {cv_results['RMSE'].std():.2f}\n"
    f"- **R²**: {cv_results['R2'].mean():.2f} ± {cv_results['R2'].std():.2f}"
)

# Prediction function for Gradio
def predict_tm_c(
    isoelectric_point, molecular_weight_da, product_conc_mg_ml, ph,
    ps80_conc, citrate_conc, llysine_conc, succinate_conc,
    kcl_conc, fructose_conc, ps50_conc, mannitol_conc,
    protein_format
):
    # Manually encode the protein format as one-hot
    formats = ["Doppelmab", "IGG3", "IGG4", "Knob-Hole", "Nano MB"]
    one_hot = [1 if protein_format == fmt else 0 for fmt in formats]

    # Create input DataFrame
    input_data = pd.DataFrame([[
        isoelectric_point, molecular_weight_da, product_conc_mg_ml, ph,
        ps80_conc, citrate_conc, llysine_conc, succinate_conc,
        kcl_conc, fructose_conc, ps50_conc, mannitol_conc,
        *one_hot  # unpack one-hot encoding
    ]], columns=[
        'isoelectric_point', 'molecular_weight_da', 'product_conc_mg_ml', 'ph',
        'ps80_conc', 'citrate_conc', 'llysine_conc', 'succinate_conc',
        'kcl_conc', 'fructose_conc', 'ps50_conc', 'mannitol_conc',
        'protein_format_doppelmab', 'protein_format_igg3',
        'protein_format_igg4', 'protein_format_knob_hole',
        'protein_format_nano_mb'
    ])

    # Predict Tm
    prediction = model.predict(input_data)[0]
    return f"Predicted Tm (°C): {prediction:.2f}"

# Gradio input components
inputs = [
    gr.Slider(label="Isoelectric Point", minimum=0.0, maximum=13.3, step=0.01, value=6.0),
    gr.Slider(label="Molecular Weight (Da)", minimum=0.0, maximum=1060459.92, step=1000, value=400000),
    gr.Slider(label="Product Concentration (mg/ml)", minimum=0.0, maximum=1036.8, step=1.0, value=258.0),
    gr.Slider(label="pH", minimum=0.0, maximum=14.0, step=0.1, value=5.2),
    gr.Slider(label="KCl Concentration", minimum=0.0, maximum=270.0, step=1.0, value=24.0),
    gr.Slider(label="Fructose Concentration", minimum=0.0, maximum=270.0, step=1.0, value=67.0),
    gr.Slider(label="Succinate Concentration", minimum=0.0, maximum=30.0, step=0.1, value=0.0),
    gr.Slider(label="L-Lysine Concentration", minimum=0.0, maximum=270.0, step=1.0, value=67.0),
    gr.Slider(label="Mannitol Concentration", minimum=0.0, maximum=270.0, step=1.0, value=0.0),
    gr.Slider(label="PS50 Concentration", minimum=0.0, maximum=0.8, step=0.01, value=0.3),
    gr.Slider(label="PS80 Concentration", minimum=0.0, maximum=0.8, step=0.01, value=0.0),
    gr.Slider(label="Citrate Concentration", minimum=0.0, maximum=30.0, step=0.1, value=0.0),

    # Dropdown for protein format selection
    gr.Dropdown(
        choices=["Doppelmab", "IGG3", "IGG4", "Knob-Hole", "Nano MB"],
        label="Protein Format",
        value="IGG3"
    ),
]

# Gradio Interface
demo = gr.Interface(
    fn=predict_tm_c,
    inputs=inputs,
    outputs="text",
    title="Tm Prediction Dashboard",
    description=validation_text
)

# Launch the app with a shareable link
demo.launch(share=True)
