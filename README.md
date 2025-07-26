This project demonstrates how to fit both a Linear Regression and a Polynomial Regression model to synthetic data using scikit-learn. It visualizes and compares how each model performs when fitting a nonlinear relationship.

🧠 Concepts Covered
Linear Regression on nonlinear data

Polynomial feature transformation

Overfitting and underfitting visualization

Model evaluation using R² score

Use of Pipeline in scikit-learn for clean workflows

📁 Files
File	Description
main.py	Full implementation of linear and polynomial regression, visualization, and comparison
README.md	Explanation of the project

📦 Requirements
Make sure you have the following libraries installed:

bash
Copy
Edit
pip install numpy matplotlib scikit-learn
🚀 How It Works
Data Generation
A synthetic quadratic function is created:

𝑦
=
0.5
𝑥
2
+
1.5
𝑥
+
2
+
noise
y=0.5x 
2
 +1.5x+2+noise
to simulate real-world nonlinear data.

Linear Regression
A basic linear model is trained on this data. Since the data is curved, the linear model underfits.

Polynomial Regression
The features are transformed using PolynomialFeatures(degree=2) to model a curve. This significantly improves performance.

Evaluation

Models are compared using R² score

Visualization helps understand which model fits better

Reusable Function
A general-purpose poly_regression(degree) function allows quick experimentation with different polynomial degrees.

📊 Output Example

📌 Example Results
lua
Copy
Edit
R² score (Linear Regression): 0.75
R² score (Polynomial Regression): 0.97
Polynomial Coefficients: [[1.5, 0.5]]
Polynomial Intercept: [2.1]
🔁 Try It Yourself
You can try different degrees like this:

python
Copy
Edit
poly_regression(1)  # Linear
poly_regression(2)  # Quadratic
poly_regression(5)  # More complex (risk of overfitting)
📚 Learning Outcome
This project is a practical example of:

Why linear models are not always enough

How feature engineering (e.g., adding polynomial terms) can drastically improve performance

Using scikit-learn's Pipeline to streamline ML workflows

👨‍💻 Author
Aviroop Ghosh
B.Tech CSE (AI) Student | Aspiring Machine Learning Engineer
