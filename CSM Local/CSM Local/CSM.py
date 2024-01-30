from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from utilsCSM import *

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_csv_file(filename):
    return pd.read_csv(filename)


@app.route('/train_evaluate', methods=['POST'])
def train_evaluate():
    if request.method == 'POST' and request.content_type.startswith('application/json'):
        data = request.get_json()
        datasets = data.get('datasets')
        task_type = request.args.get('type')  # Get the task type from the query parameter

        if not datasets or len(datasets) == 0:
            return jsonify({'error': 'No datasets selected for training and evaluation.'})

        results = []
        for dataset in datasets:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset)
            X, y = preprocess_regression_classification(file_path)

            if task_type == 'classification':
                accuracy, recall, precision, f1 = train_classification_model(X, y)
                result = {
                    'filename': dataset,
                    'accuracy': accuracy,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1
                }
            elif task_type == 'regression':
                r_squared, adj_r_squared, mean_abs_error = train_regression_model(X, y)
                result = {
                    'filename': dataset,
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'mean_abs_error': mean_abs_error
                }
            elif task_type == 'clustering':
                X, y = preprocess_clustering(file_path)
                optimal_clusters, silhouette_score_optimal, ari_optimal = evaluate_clustering_model(X, y)
                result = {
                    'filename': dataset,
                    'optimal_clusters': optimal_clusters,
                    'silhouette_score': silhouette_score_optimal,
                    'ari': ari_optimal
                }
            else:
                return jsonify({'error': 'Invalid task type.'})

            results.append(result)

        # Calculate the differences in metrics for each result
        differences = []
        for i in range(len(results)):
            if i > 0:
                difference = {}
                for key in results[i].keys():
                    if key != 'filename':
                        diff_value = abs(results[i][key] - results[i - 1][key])
                        difference[key] = diff_value
                differences.append(difference)

        return jsonify({'results': results, 'differences': differences})

    return jsonify({'error': 'Invalid request. Expected JSON data.'})

@app.route('/result')
def result():
    selected_datasets = request.args.get('datasets')
    task_type = request.args.get('type')  # Get the task type from the query parameter

    if selected_datasets:
        datasets = selected_datasets.split(',')
        results = fetch_results_from_datasets(datasets)

        # Separate the results based on the task type
        classification_results = []
        regression_results = []
        clustering_results = []

        for result in results:
            if task_type == 'classification':
                classification_results.append(result)
            elif task_type == 'regression':
                regression_results.append(result)
            elif task_type == 'clustering':
                clustering_results.append(result)

        # Calculate the differences in metrics for each result
        differences = []
        for i in range(len(results)):
            if i > 0:
                difference = {}
                for key in results[i].keys():
                    if key != 'filename':
                        diff_value = abs(results[i][key] - results[i - 1][key])
                        difference[key] = diff_value
                differences.append(difference)

        return render_template('result.html', classification_results=classification_results,
                               regression_results=regression_results, clustering_results=clustering_results,
                               differences_classification=differences[:len(classification_results) - 1],
                               differences_regression=differences[len(classification_results):len(classification_results) + len(regression_results) - 1],
                               differences_clustering=differences[len(classification_results) + len(regression_results):])
    else:
        return redirect(url_for('CSM'))

@app.route('/', methods=['GET', 'POST'])
def CSM():
    filenames = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('CSM.html')

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    filenames = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('evaluate.html', filenames=filenames)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print("Upload Function working.")
    if request.method == 'POST' and 'csv_file' in request.files:
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file.'})

        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            df = read_csv_file(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            top_5_rows = df.head(5).to_html(index=False)
            return jsonify({'message': 'File uploaded successfully.', 'top_5_rows': top_5_rows})
    
        return jsonify({'error': 'Invalid file format. Only CSV files are allowed.'})
    else:
        clear_upload_folder()
    return render_template('upload.html')

@app.route('/view', methods=['GET', 'POST'])
def view_csv():
    if request.method == 'POST':
        filename = None
        if request.content_type.startswith('application/json'):
            # If the content type is JSON, retrieve the filename from the JSON data
            filename = request.json.get('filename')
        elif request.content_type.startswith('multipart/form-data'):
            # If the content type is form data, retrieve the filename from the form data
            filename = request.form.get('filename')

        if filename is None:
            return jsonify({'error': 'Invalid request. Missing filename.'})

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            df = read_csv_file(file_path)
            columns = df.columns.tolist()
            rows = df.to_dict(orient='records')
            return jsonify({'columns': columns, 'rows': rows})

        return jsonify({'error': 'File not found.'})

    # Add a response for GET requests (if needed)
    return jsonify({'message': 'This endpoint supports POST requests only.'})



@app.route('/edit', methods=['POST'])
def edit_column():
    filename = request.form['filename']
    column = request.form['column']
    action = request.form['action']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        df = read_csv_file(file_path)

        if action == 'prediction':
            # Append column name with "DV" at the end of the string
            new_column_name = f"{column} DV"
            df[new_column_name] = df[column]  # Replace with your prediction logic
            df = df.drop(column, axis=1)

        df.to_csv(file_path, index=False)

        top_5_rows = df.head(5).to_html(index=False)
        return jsonify({'message': f'{column} tagged as Target Column (Dependent Variable).',
                        'top_5_rows': top_5_rows})

    return jsonify({'error': 'File not found.'})

if __name__ == '__main__':
    app.run(debug=True)