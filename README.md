# interactive-visualization-system-for-identifying-the-impacting-factors-of-diabetic

To run this one, follow these steps:
  1. Install Python
  2. Run "Visualizer.dat"
  3. A tab will open in the browser
  4. Upload the desired dataset to analyze


# An updated version of this application is on the way, where more functionality will be added
How the System Works
The Interactive Visualization System you have built works like this:

Upload Data:

The user uploads a .csv or .xlsx file containing diabetes (or any other numeric) data.

The system reads the file and shows a preview.

Filtering the Data:

Classic Filter: A basic filter where you choose one numeric column and select a value range.

Dynamic Filter: Advanced filtering where you can filter multiple columns dynamically based on sliders (for numbers) or multi-selection (for categories).

Data Visualization:

Correlation Heatmap: Shows how strong the relationships are between numeric variables (using Pearson’s correlation).

Scatter Plot: Shows relationships between two features, colored by a third feature (which could be categorical or numeric).

Machine Learning Insights:

Feature Importance (Random Forest Model):

The app trains a Random Forest Classifier model.

Calculates how important each feature is in predicting the target variable (which could be 'Outcome' or another column).

Displays a bar chart of feature importance and model accuracy.

Clustering Analysis:

Unsupervised learning to find hidden groups/patterns in data:

K-Means clustering (you choose how many clusters you want).

DBSCAN clustering (you set parameters like epsilon and min_samples).

Output:

Data tables

Graphs (scatterplots, heatmaps)

Feature importance scores

Cluster labels

Model accuracy score


