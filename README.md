# 🎓 Student Performance Prediction Using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning-based web application designed to predict students' academic performance based on their study habits, previous scores, extracurricular activities, sleeping hours, and sample question papers practiced.

This project uses **Linear Regression**, a supervised Machine Learning algorithm, to analyze student-related information and predict their Performance Index.

The application is developed using Python, Flask, HTML, CSS, Pandas, NumPy, and Scikit-learn. Users can enter their academic details through a user-friendly web interface and receive an estimated performance score.

### 🎯 Project Objectives

- Analyze student academic performance using historical data.
- Understand the relationship between study habits and performance.
- Apply Exploratory Data Analysis (EDA) techniques.
- Build a Linear Regression prediction model.
- Evaluate the model using R² Score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- Develop an interactive web application using Flask.
- Predict student performance based on user-provided inputs.

---

## 📊 Dataset Information

**Dataset Name:** `Student_Performance.csv`

The dataset contains 10,000 student records and 6 columns, including 5 independent variables and 1 dependent variable.

### Dataset Features

| Feature | Description | Type |
|---|---|---|
| Hours Studied | Number of hours spent studying | Numerical |
| Previous Scores | Student's previous examination scores | Numerical |
| Extracurricular Activities | Participation in extracurricular activities | Categorical |
| Sleep Hours | Number of hours spent sleeping | Numerical |
| Sample Question Papers Practiced | Number of sample papers practiced | Numerical |
| Performance Index | Student's performance score | Target |

### Dataset Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Display the first five records.
3. Check dataset dimensions and data types.
4. Identify missing values.
5. Convert categorical values into numerical values.

**Categorical Encoding:**

- Yes → 0
- No → 1

**Independent Variables (X):** First five columns.

**Dependent Variable (y):** Performance Index.

### Dataset Splitting

The dataset is divided into training and testing sets.

- Training Data: 80%
- Testing Data: 20%
- Random State: 42

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computations |
| Pandas | Dataset manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning model |
| Linear Regression | Prediction algorithm |
| Flask | Backend web framework |
| HTML5 | Webpage structure |
| CSS3 | Webpage styling |
| Pickle | Model serialization |
| Gunicorn | Production application server |

---

## ⚙️ Installation Steps

Follow these steps to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Student-Performance-Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Student-Performance-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Linux / macOS:

```bash
source venv/bin/activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Train the Machine Learning Model

```bash
python main.py
```

This executes the following operations:

- Dataset loading and inspection.
- Data preprocessing.
- Training and testing data splitting.
- Linear Regression model training.
- Model evaluation.
- Model serialization using Pickle.

The trained model is saved as:

```text
model.pkl
```

### Step 7: Run the Flask Application

```bash
python app.py
```

### Step 8: Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

Enter the student details and click the **Predict Performance** button to obtain the predicted Performance Index.

---

## 📁 Project Structure

```text
Student-Performance-Prediction/
│
├── main.py
│   └── Machine Learning model development
│
├── app.py
│   └── Flask application and prediction logic
│
├── Student_Performance.csv
│   └── Student performance dataset
│
├── model.pkl
│   └── Trained Linear Regression model
│
├── requirements.txt
│   └── Required Python dependencies
│
├── Procfile
│   └── Deployment server configuration
│
├── templates/
│   └── index.html
│       └── User interface
│
└── README.md
    └── Project documentation
```

---

## 🤖 Model Training Process

### 1. Data Loading

The Student Performance dataset is loaded using Pandas.

```python
self.df = pd.read_csv('Student_Performance.csv')
```

### 2. Data Preprocessing

The categorical feature Extracurricular Activities is converted into numerical values using mapping.

```python
self.df['Extracurricular Activities'] = self.df[
    'Extracurricular Activities'
].map({'Yes': 0, 'No': 1})
```

### 3. Feature Selection

The independent and dependent variables are separated.

```python
self.X = self.df.iloc[:, :-1]
self.y = self.df.iloc[:, -1]
```

### 4. Train-Test Splitting

```python
self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    self.X,
    self.y,
    test_size=0.2,
    random_state=42
)
```

### 5. Model Training

Linear Regression is used to train the model.

```python
self.reg = LinearRegression()
self.reg.fit(self.X_train, self.y_train)
```

### 6. Model Prediction

The trained model generates predictions for both training and testing datasets.

```python
self.train_predictions = self.reg.predict(self.X_train)

self.test_predictions = self.reg.predict(self.X_test)
```

### 7. Model Evaluation

The model performance is evaluated using:

- **R² Score:** Measures how well the model explains variations in the target variable.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the target variable's original units.

### 8. Model Saving

The trained model is saved using Pickle.

```python
with open("model.pkl", "wb") as f:
    pickle.dump(self.reg, f)
```

---

## 📈 Results and Model Performance

The trained Linear Regression model was evaluated using an 80:20 training and testing split with random state 42.

### Model Evaluation Results

| Evaluation Metric | Training Data | Testing Data |
|---|---:|---:|
| R² Score | 0.9887 | 0.9890 |
| MSE | 4.1697 | 4.0826 |
| RMSE | 2.0420 | 2.0206 |

### Performance Summary

- Training R² Score: **98.87%**
- Testing R² Score: **98.90%**
- Training RMSE: **2.04**
- Testing RMSE: **2.02**

The model demonstrates a strong fit on both training and testing data, with similar evaluation results.

---

## 🌐 Deployment

The Flask application is configured to run using Gunicorn.

### Deployment Configuration

The Procfile contains:

```text
web: gunicorn app:app
```

### Deployment Steps

1. Upload the project files to a GitHub repository.
2. Connect the GitHub repository to a compatible Python hosting platform, such as Render.
3. Configure the build command:

```bash
pip install -r requirements.txt
```

4. Configure the start command:

```bash
gunicorn app:app
```

5. Deploy the application.
6. Open the generated application URL to access the Student Performance Prediction system.

**Note:** The deployment configuration is included in the project. A live deployment URL should be added after successful deployment.
## Deployment URL : https://student-performance-yaj2.onrender.com
