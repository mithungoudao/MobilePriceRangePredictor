import os
import pandas as pd
import joblib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Define the upload folder and ensure it exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Check if the file is part of the request
        if 'file' not in request.files:
            return "No file part in the request"

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            return "No selected file"

        # If a file is selected, process it
        if file:
            # Save the file to the 'uploads' directory
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Run the Jupyter Notebook to retrain the model
            run_notebook(file_path)

            return redirect(url_for("index"))  # Redirect back to the home page after upload

    # Render the upload form if request is GET
    return render_template("upload.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect the form data and create a DataFrame
        new_data = {
            'battery_power': [float(request.form['battery_power'])],
            'blue': [int(request.form['blue'])],
            'clock_speed': [float(request.form['clock_speed'])],
            'dual_sim': [int(request.form['dual_sim'])],
            'fc': [float(request.form['fc'])],
            'four_g': [int(request.form['four_g'])],
            'int_memory': [int(request.form['int_memory'])],
            'm_dep': [float(request.form['m_dep'])],
            'mobile_wt': [int(request.form['mobile_wt'])],
            'n_cores': [int(request.form['n_cores'])],
            'pc': [float(request.form['pc'])],
            'px_height': [int(request.form['px_height'])],
            'px_width': [int(request.form['px_width'])],
            'ram': [int(request.form['ram'])],
            'sc_h': [float(request.form['sc_h'])],
            'sc_w': [float(request.form['sc_w'])],
            'talk_time': [int(request.form['talk_time'])],
            'three_g': [int(request.form['three_g'])],
            'touch_screen': [int(request.form['touch_screen'])],
            'wifi': [int(request.form['wifi'])]
        }

        # Convert new data to DataFrame and make predictions
        new_df = pd.DataFrame(new_data)

        # Load the trained model and predict the price range
        model = joblib.load('model.pkl')
        price_range_prediction = model.predict(new_df)

        # Return the prediction result
        return render_template("result.html", prediction=int(price_range_prediction[0]))

    return render_template("predict.html")

def run_notebook(file_path):
    # Load the notebook
    notebook_file = 'mobile.ipynb'
    
    with open(notebook_file) as f:
        nb = nbformat.read(f, as_version=4)

    # Prepare the notebook to be executed
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Execute the notebook
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_file)}})
    except Exception as e:
        print(f"Error running notebook: {e}")

    # Optionally save the notebook or model here if needed
    # You can also overwrite the notebook if desired
    with open(notebook_file, 'w') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    app.run(debug=True)