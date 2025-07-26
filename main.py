import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# Generate quadratic data with noise
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + 1.5 * X + 2 + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression (baseline)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_linear = lin_reg.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

# Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print metrics
print("R² score (Linear Regression):", r2_linear)
print("R² score (Polynomial Regression):", r2_poly)
print("Polynomial Coefficients:", poly_reg.coef_)
print("Polynomial Intercept:", poly_reg.intercept_)

# Plot comparison
X_curve = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
X_curve_poly = poly.transform(X_curve)
y_curve_poly = poly_reg.predict(X_curve_poly)

plt.scatter(X, y, color='lightgreen', label='Original data')
plt.plot(X_curve, y_curve_poly, color='purple', label='Polynomial Regression (degree=2)')
plt.plot(X_test, y_pred_linear, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

# Generalized function for any degree
def poly_regression(degree=2):
    X_new = np.linspace(-3, 3, 200).reshape(200, 1)
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_reg = LinearRegression()

    poly_regression_pipeline = Pipeline([
        ('poly_features', poly_features),
        ('lin_reg', lin_reg)
    ])

    poly_regression_pipeline.fit(X_train, y_train)
    y_new = poly_regression_pipeline.predict(X_new)

    plt.scatter(X, y, color='lightgray', label='Original Data')
    plt.plot(X_new, y_new, label=f'Polynomial Regression (degree={degree})', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression Curve (degree={degree})')
    plt.legend()
    plt.axis([-4, 4, 0, 12])
    plt.grid(True)
    plt.show()

# # Try different degrees
# poly_regression(1)
# poly_regression(2)
poly_regression(5)
poly_regression(10)