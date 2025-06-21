import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Modell laden
model = joblib.load('C:/Users/hempe/Studium/Real_Project/Project_repo/models/gradient_boosting_model.pkl')

# Load validation results
cv_results = pd.read_csv(r'C:/Users/hempe/Studium/Real_Project/Project_repo/models/validation_results.csv')

# Validierungsergebnisse vorbereiten
validation_text = (
    f"### 🔍 Modellvalidierung (5-Fold-CV)\n"
    f"- **RMSE**: {cv_results['RMSE'].mean():.2f} ± {cv_results['RMSE'].std():.2f}\n"
    f"- **R²**: {cv_results['R2'].mean():.2f} ± {cv_results['R2'].std():.2f}"
)


# Vorhersagefunktion
def predict_tm_c(
    isoelectric_point, molecular_weight_da, product_conc_mg_ml, ph,
       ps80_conc, citrate_conc, llysine_conc, succinate_conc,
       kcl_conc, fructose_conc, ps50_conc, mannitol_conc,
       protein_format_doppelmab, protein_format_igg3,
       protein_format_igg4, protein_format_knob_hole,
       protein_format_nano_mb
):
    input_data = pd.DataFrame([[  # 2D-Array mit einer Zeile
       isoelectric_point, molecular_weight_da, product_conc_mg_ml, ph,
       ps80_conc, citrate_conc, llysine_conc, succinate_conc,
       kcl_conc, fructose_conc, ps50_conc, mannitol_conc,
       protein_format_doppelmab, protein_format_igg3,
       protein_format_igg4, protein_format_knob_hole,
       protein_format_nano_mb
    ]], columns=[
        'isoelectric_point', 'molecular_weight_da', 'product_conc_mg_ml', 'ph',
       'ps80_conc', 'citrate_conc', 'llysine_conc', 'succinate_conc',
       'kcl_conc', 'fructose_conc', 'ps50_conc', 'mannitol_conc',
       'protein_format_doppelmab', 'protein_format_igg3',
       'protein_format_igg4', 'protein_format_knob_hole',
       'protein_format_nano_mb'
    ])
    
    prediction = model.predict(input_data)[0]
    return f"Predicted Tm (°C): {prediction:.2f}"

# Gradio Interface
inputs = [
    gr.Slider(label="Isoelectric Point", minimum=0.0, maximum=13.3, step=0.01, value=6.0),              
    gr.Slider(label="Molecular Weight (Da)", minimum=0.0, maximum=1060459.92, step=1000, value=400000), 
    gr.Slider(label="Product Concentration (mg/ml)", minimum=0.0, maximum=1036.8, step=1.0, value=258.0),
    gr.Slider(label="pH", minimum=0.0, maximum=14.0, step=0.1, value=5.2),                               
    gr.Slider(label="KCl Concentration", minimum=0.0, maximum=270.0, step=1.0, value=24.0),              
    gr.Slider(label="Fructose Concentration", minimum=0.0, maximum=540.0, step=1.0, value=67.0),         
    gr.Slider(label="Succinate Concentration", minimum=0.0, maximum=30.0, step=0.1, value=0.0),          
    gr.Slider(label="L-Lysine Concentration", minimum=0.0, maximum=270.0, step=1.0, value=67.0),         
    gr.Slider(label="Mannitol Concentration", minimum=0.0, maximum=540.0, step=1.0, value=0.0),          
    gr.Slider(label="PS50 Concentration", minimum=0.0, maximum=0.8, step=0.01, value=0.3),               
    gr.Slider(label="PS80 Concentration", minimum=0.0, maximum=0.8, step=0.01, value=0.0),               
    gr.Slider(label="Citrate Concentration", minimum=0.0, maximum=30.0, step=0.1, value=0.0),    
    gr.Radio([0, 1], label="Protein Format Doppelmab", value=0),        
    gr.Radio([0, 1], label="Protein Format IGG3", value=1),  
    gr.Radio([0, 1], label="Protein Format IGG4", value=0),
    gr.Radio([0, 1], label="Protein Format Knob-Hole", value=0),
    gr.Radio([0, 1], label="Protein Format Nano MB", value=0),
    
]

# UI-Komponenten
demo = gr.Interface(
    fn=predict_tm_c,
    inputs=inputs,
    outputs="text",
    title="Tm Prediction Dashboard",
    description=validation_text  # Statische Anzeige oberhalb des Dashboards
)
demo.launch(share=True)

