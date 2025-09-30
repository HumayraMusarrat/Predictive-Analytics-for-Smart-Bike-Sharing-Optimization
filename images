# visualize_results.py
import matplotlib.pyplot as plt
import seaborn as sns

# Example model performance data
models = ['Random Forest', 'KNN', 'Logistic Regression', 'LDA', 'QDA', 'Naive Baseline']
accuracy = [93, 92, 90, 90, 90, 50]
f1_weighted = [92, 92, 89, 90, 90, 50]
f1_macro = [86, 86, 84, 85, 85, 50]

# Accuracy Comparison
plt.figure(figsize=(10,6))
sns.barplot(x=models, y=accuracy, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig("images/model_accuracy.png")
plt.close()

# F1-Weighted Comparison
plt.figure(figsize=(10,6))
sns.barplot(x=models, y=f1_weighted, palette="magma")
plt.title("F1-Weighted Score Comparison")
plt.ylabel("F1-Weighted (%)")
plt.ylim(0, 100)
plt.savefig("images/f1_weighted.png")
plt.close()

# F1-Macro Comparison
plt.figure(figsize=(10,6))
sns.barplot(x=models, y=f1_macro, palette="coolwarm")
plt.title("F1-Macro Score Comparison")
plt.ylabel("F1-Macro (%)")
plt.ylim(0, 100)
plt.savefig("images/f1_macro.png")
plt.close()
