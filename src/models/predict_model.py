import gradio as gr
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns

# Set the MLflow experiment name
mlflow.set_experiment("tm_prediction_experiment")

# Load the latest run ID from a file
with open("latest_run.txt", "r") as f:
    run_id = f.read().strip()

# Load the trained model from MLflow using the run ID
model_uri = f"runs:/{run_id}/gradient_boosting_model"
model = mlflow.sklearn.load_model(model_uri)

# Load valudation data from metrics of the same run
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
metrics = run.data.metrics

rmse = metrics.get("rmse")
r2 = metrics.get("r2")

# Format the cross-validation results for display
validation_text = (
    f"### 🔍 Model Validation\n"
    f"- **RMSE**: {rmse:.2f}\n"
    f"- **R²**: {r2:.2f}"
)

# Function to generate and save feature importance plot
def generate_feature_importance_plot():
    importances = model.feature_importances_
    features = model.feature_names_in_

    feat_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='Blues_d')
    plt.title('Feature Importance - GradientBoostingRegressor')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plot_path = "data/processed/feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saving feature importance plot to: {plot_path}")
    return plot_path



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

    # Ensure the input DataFrame has the same columns as the model expects
    input_data = input_data[model.feature_names_in_]

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

# Build Gradio App with Tabs
with gr.Blocks() as demo:
    with gr.Tab("🔬 Predict Tm"):
        gr.Markdown(validation_text)
        output = gr.Textbox(label="Prediction Result")
        gr.Interface(fn=predict_tm_c, inputs=inputs, outputs=output)

    with gr.Tab("📊 Feature Importance"):
        plot_button = gr.Button("Generate Feature Importance Plot")
        plot_output = gr.Image(type="filepath")

        # Button-Callback
        plot_button.click(fn=generate_feature_importance_plot, outputs=plot_output)


# Launch the app with a shareable link
demo.launch(share=True)

