# Product Categorizer

A standalone Streamlit seller utility that uses the existing trained category and subcategory models to help identify the most relevant ecommerce listing category. The original `.pkl` files are loaded locally and are not retrained or modified.

## First ML model and training data

The first version of the classifier was trained on the Alifeo e-commerce product categorization dataset, containing 100K+ product records. Using more product examples helps the model learn a wider range of product titles, descriptions, categories, and subcategories.

The training pipeline cleans the product text, creates TF-IDF features, and trains separate LinearSVC models for category and subcategory prediction. The trained models and vectorizers are saved as `.pkl` files and loaded by the Streamlit app.

- Kaggle dataset: [Alifeo E-commerce Product Categorization Dataset](https://www.kaggle.com/datasets/alifeo/alifeo-e-commerce-product-categorization-dataset)
- Google Colab training notebook: [Open `ecommerce.ipynb` in Google Colab](https://colab.research.google.com/github/theYahyaturk/category/blob/main/ecommerce.ipynb)
- Local notebook file: [`ecommerce.ipynb`](ecommerce.ipynb)
live : https://category-ers8pwt9x3frhukcral63t.streamlit.app/

The current deployed app uses the saved model artifacts; it does not retrain the models during startup.

## Project structure

```text
alifeo-ml-classifier/
├── models/
│   ├── category_model.pkl
│   ├── category_vectorizer.pkl
│   ├── subcategory_model.pkl
│   └── subcategory_vectorizer.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally on Windows

Open PowerShell in this folder and create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`. Open that address in a browser to use Product Categorizer.

## Deploy later on Streamlit Community Cloud

1. Create a GitHub repository and upload this project, including the files in `models/`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, choose the repository and branch, and set the main file to `app.py`.
4. Deploy the app. Streamlit Cloud installs the packages from `requirements.txt` and runs the app automatically.

The model files are large, so confirm that the repository and GitHub account can store them before deploying. Never commit secrets; this demo does not require any.
