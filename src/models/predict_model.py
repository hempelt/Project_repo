import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Modell und Scaler laden
model = joblib.load('C:/Users/hempe/Studium/Real_Project/Project_repo/models/gradient_boosting_model.pkl')
scaler = joblib.load('C:/Users/hempe/Studium/Real_Project/Project_repo/models/modelsscaler.pkl')

# Vorhersagefunktion
def predict_tm_c(
    isoelectric_point, molecular_weight_da, product_conc_mg_ml, ph,
    kcl_conc, fructose_conc, succinate_conc, l_lysine_conc,
    mannitol_conc, ps50_conc, ps80_conc, citrate_conc,
    protein_format_igg3, protein_format_igg4,
    protein_format_knob_hole, protein_format_nano_mb
):
    input_data = pd.DataFrame([[
        isoelectric_point, molecular_weight_da, product_conc_mg_ml, ph,
        kcl_conc, fructose_conc, succinate_conc, l_lysine_conc,
        mannitol_conc, ps50_conc, ps80_conc, citrate_conc,
        protein_format_igg3, protein_format_igg4,
        protein_format_knob_hole, protein_format_nano_mb
    ]], columns=[
        'isoelectric_point', 'molecular_weight_da', 'product_conc_mg_ml', 'ph',
        'kcl_conc', 'fructose_conc', 'succinate_conc', 'l-lysine_conc',
        'mannitol_conc', 'ps50_conc', 'ps80_conc', 'citrate_conc',
        'protein_format_igg3', 'protein_format_igg4',
        'protein_format_knob_hole', 'protein_format_nano_mb'
    ])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return f"Predicted Tm (°C): {prediction:.2f}"

# Gradio Interface
inputs = [
    gr.Number(label="Isoelectric Point"),
    gr.Number(label="Molecular Weight (Da)"),
    gr.Number(label="Product Concentration (mg/ml)"),
    gr.Number(label="pH"),
    gr.Number(label="KCl Concentration"),
    gr.Number(label="Fructose Concentration"),
    gr.Number(label="Succinate Concentration"),
    gr.Number(label="L-Lysine Concentration"),
    gr.Number(label="Mannitol Concentration"),
    gr.Number(label="PS50 Concentration"),
    gr.Number(label="PS80 Concentration"),
    gr.Number(label="Citrate Concentration"),
    gr.Radio([0, 1], label="Protein Format IGG3"),
    gr.Radio([0, 1], label="Protein Format IGG4"),
    gr.Radio([0, 1], label="Protein Format Knob-Hole"),
    gr.Radio([0, 1], label="Protein Format Nano MB"),
]

demo = gr.Interface(fn=predict_tm_c, inputs=inputs, outputs="text", title="Tm Prediction Dashboard")
demo.launch()
