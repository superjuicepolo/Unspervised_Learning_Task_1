import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Loading File and Understanding ---------------------------------------------------------------------- Start

dt = pd.read_csv("Dataset/mental-heath-in-tech-2016_20161114.csv")
dt_rows, dt_columns = dt.shape

print("Survey Questions:","\n" + "\n".join(f"{i} {col}" for i, col in enumerate(dt.columns)))



# Plotting the counts
plt.figure(figsize=(8, 5))
sns.countplot(y=dt["Do you currently have a mental health disorder?"], order=dt["Do you currently have a mental health disorder?"].value_counts().index, palette="Set2")
plt.title("Distribution of Responses to 'Do you currently have a mental health disorder?'")
plt.xlabel("Count")
plt.ylabel("Response")
plt.tight_layout()
plt.show()


print(dt["Do you currently have a mental health disorder?"])

def show_missing_data_summary():
    missing_counts = dt.isna().sum()
    missing_list = missing_counts.tolist()
    num_columns_with_missing = sum(1 for count in missing_list if count > 0)
    total_columns = len(missing_list)
    percent_with_missing = (num_columns_with_missing / total_columns) * 100

    return f"{percent_with_missing:.2f}% of columns have missing data.\nMissing counts per column: {missing_list}"

print(show_missing_data_summary())

def show_missing_values(dataframe, min_missing_ratio=0.3):
    total_rows = len(dataframe)
    min_missing_count = total_rows * min_missing_ratio

    for column_index, column_name in enumerate(dataframe.columns):
        missing_count = dataframe[column_name].isna().sum()
        if missing_count >= min_missing_count:
            missing_percent = (missing_count / total_rows) * 100
            column_type = dataframe[column_name].dtype
            print(f"{missing_percent:.2f}% Missing -> ({column_index}/{column_type}) {column_name}")


show_missing_values(dt, min_missing_ratio=0.1)

def plot_pca_projection(dataframe, output_name, file_type='png'):


    # Step 1: Encode categorical features
    label_encoder = LabelEncoder()
    numeric_data = dataframe.apply(lambda col: label_encoder.fit_transform(col))

    # Step 2: Standardize the dataset
    standardizer = StandardScaler()
    standardized_data = standardizer.fit_transform(numeric_data)

    # Step 3: Apply PCA to reduce to 2D
    pca_model = PCA(n_components=2)
    pca_components = pca_model.fit_transform(standardized_data)

    # Step 4: Plot and save
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_components[:, 0], pca_components[:, 1], alpha=0.5)
    ax.set_title('2D PCA Projection')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.grid(True)

    save_path = f"{output_name}.{file_type}"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

plot_pca_projection(dt, output_name="pca_plot_after_loading")

# Loading File and Understanding ---------------------------------------------------------------------- Finish

# Preproccesing ---------------------------------------------------------------------- Start

print(dt[dt['What is your age?'] > 99])
print(dt["What is your age?"][564])
print(dt[dt['What is your age?']<17])
print(dt["What is your age?"][93])
print(dt["What is your age?"][656])
print(dt["What is your age?"][808])

age_col = "What is your age?"

# Find incorrect age rows
out_of_range = (dt[age_col] < 17) | (dt[age_col] > 99)
bad_age_rows = dt[out_of_range]

print(f'Found {len(bad_age_rows)} incorrect values in "{age_col}".')

# Get only valid age rows
good_age_rows = dt[dt[age_col].between(17, 99)]

# Compute integer mean from valid values
average_age = int(good_age_rows[age_col].mean())

# Fix the bad values
dt.loc[out_of_range, age_col] = average_age

print(f"Replaced invalid ages at indices {bad_age_rows.index.tolist()} with mean age ({average_age}).")

print(dt["What is your gender?"])
print(dt[dt['What is your gender?']==None])
print(dt["What is your gender?"].value_counts())
print(dt["What is your gender?"].isna().sum())
print(dt["What is your gender?"].unique())
gender_male_list=['Male', 'male', 'Male ', 'M', 'm','man',
  'Male.' , 'Male (cis)' , 'Other' , 'nb masculine' , 'Man'
 'Sex is male' , 'cis male' , 'Dude' ,
 "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? "
 'mail' , 'M|' , 'male ' , 'Cis Male' , 'cisdude' ,'cis man' ,  'MALE']

gender_female_list=["Female", "female", "I identify as female.", "female ", 'Female assigned at birth ', 'F', 'Woman'
, 'fm', 'f', 'Cis female ' ,  'Female ' , 'woman' , 'female/woman',
  'Cisgender Female' ,
  'fem' ,'Female (props for making this a freeform field, though)' , ' Female'
 , 'Cis-woman' ,]

def define_gender(value):
    print(value)
    if value in gender_male_list:
        return "Male"
    elif value in gender_female_list:
        return "Female"
    else:
        return "Other"



dt['What is your gender?'] = dt['What is your gender?'].apply(define_gender)


print(dt["How many employees does your company or organization have?"].value_counts())
print(dt["How many employees does your company or organization have?"].isnull())
print(dt["How many employees does your company or organization have?"].isna().sum())

dt=dt.loc[(dt['Are you self-employed?'] == 0)].reset_index(drop=True)

# Could do: add line when the column will not have missing values
# dt=dt.loc[(dt["Is your employer primarily a tech company/organization?"]==1.0)].reset_index(drop=True)

# Column name where diagnosis info is stored
diagnosis_column = 'If yes, what condition(s) have you been diagnosed with?'

# Display value counts for manual inspection (optional)
print(dt[diagnosis_column].value_counts())

# Mapping for identifying key terms within the text entries
condition_flags = {
    'AnxDis': 'Anxiety',
    'MoodDis': 'Mood',
    'AttentionProb': 'Attention',
    'ObsessiveTraits': 'Compulsive',
    'PostTrauma': 'Post',
    'UnconfirmedPTSD': 'PTSD \(undiagnosed\)',
    'EatingBehav': 'Eating',
    'SubstanceDep': 'Substance',
    'StressSynd': 'Stress Response',
    'PersType': 'Personality',
    'DevDis': 'Pervasive',
    'Psychosis': 'Psychotic',
    'AddictivePat': 'Addictive',
    'IdentitySplit': 'Dissociative',
    'SeasonalMood': 'Seasonal',
    'SchizoType': 'Schizotypal',
    'TBI': 'Brain',
    'SexCompuls': 'Sexual',
    'NeuroType': 'Autism',
    'ADDOnly': 'ADD \(w/o Hyperactivity\)'
}

# Apply text search for each condition and generate boolean flags
for flag, pattern in condition_flags.items():
    dt[flag] = dt[diagnosis_column].str.contains(pattern, regex=True, case=False).fillna(False).astype(bool)

# Better to think about Nan values for the column 'If yes, what condition(s) have you been diagnosed with?'

dt["What US state or territory do you live in?"]=dt["What US state or territory do you live in?"].fillna("Nane")
dt["What US state or territory do you work in?"]=dt["What US state or territory do you work in?"].fillna("Nane")

# "What country do you live in?"
# "What country do you work in?"

print("\n".join([col for col in dt.columns if dt[col].nunique() > 10]))
dt.drop(columns=["Why or why not?",
"Why or why not?.1",
"If yes, what condition(s) have you been diagnosed with?",
"If maybe, what condition(s) do you believe you have?",
"If so, what condition(s) were you diagnosed with?"
],inplace=True)

dt.drop(columns=['Are you self-employed?',
       'Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?',
       'Do you know local or online resources to seek help for a mental health disorder?',
       'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?',
       'If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?',
       'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?',
       'If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?',
       'Do you believe your productivity is ever affected by a mental health issue?',
       'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?'
],inplace=True)

dt.to_csv("dataset_after_processing_v1.csv", index=False)

# Preproccesing ---------------------------------------------------------------------- Finish

# Filling Nan values  ---------------------------------------------------------------------- Start

import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OrdinalEncoder

print("Before Imputation:\n", dt)

# Step 1: Identify categorical columns
categorical_cols = dt.select_dtypes(include=['object', 'category']).columns.tolist()

# Step 2: Encode categorical columns and store mapping
encoder = OrdinalEncoder()
dt_encoded = dt.copy()
dt_encoded[categorical_cols] = encoder.fit_transform(dt[categorical_cols])

# Create mappings for each categorical column
category_mappings = {
    col: dict(zip(classes, range(len(classes))))
    for col, classes in zip(categorical_cols, encoder.categories_)
}

print("\nCategory Mappings:")
for col, mapping in category_mappings.items():
    print(f"{col}: {mapping}")

# Step 3: Impute missing values using Bayesian Ridge
imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
dt_imputed_array = imputer.fit_transform(dt_encoded)
dt_imputed = pd.DataFrame(dt_imputed_array, columns=dt.columns)

# Optional Step 4: Round categorical values and decode back
# for col in categorical_cols:
    # dt_imputed[col] = dt_imputed[col].round().astype(int)
    # inverse_map = dict(enumerate(encoder.categories_[categorical_cols.index(col)]))
    # dt_imputed[col] = dt_imputed[col].map(inverse_map)

print("\nImputed Data:\n", dt_imputed)
dt_imputed.to_csv("dataset_after_filling_Nans.csv", index=False)
# Filling Nan values  ---------------------------------------------------------------------- Finish

# Selecting features  ---------------------------------------------------------------------- Start

import seaborn as sns
import matplotlib.pyplot as plt

matrix = dt_imputed.corr()
matrix.to_csv("correlation_output.csv", index=False)

fig_width, fig_height = 36, 24
color_scheme = 'coolwarm'

plt.figure(figsize=(fig_width, fig_height))
sns.heatmap(matrix, annot=True, cmap=color_scheme, fmt=".1f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig("correlation_plot.png", bbox_inches='tight')
plt.close()


import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

column_y = 'Do you currently have a mental health disorder?'
X = dt_imputed.drop(column_y, axis=1)
y = dt_imputed[[column_y]].values.ravel()
X = StandardScaler().fit_transform(X)

svm = SVC(kernel='linear')

sfs = SFS(svm, k_features="best", forward=True, floating=False, scoring='accuracy', cv=5, n_jobs=-1)
sfs.fit(X, y)

print("\nSelected Feature Names:")
print(sfs.k_feature_names_)

sfs_results = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sfs_results = sfs_results.sort_values(by='avg_score', ascending=False)
sfs_results.index.name = 'Feature Index Set'
sfs_results.reset_index(inplace=True)
print("\nTop SFS Results: \n" + sfs_results[['Feature Index Set', 'avg_score', 'std_dev', 'feature_names']].head(10).to_string(index=False))
sfs_results.to_csv("sfs_results.csv", index = False)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Feature indices to keep

# Average score = 86.471 % / Accuracy after training the model = 86.522 %
# fi = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 22, 25, 26, 27, 28, 32, 35, 38, 39, 40, 42, 43, 44, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
# Average score = 85.948 % / Accuracy after training the model = 86.957 %
fi = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 26, 27, 28, 35, 38, 39, 42, 44, 48, 49, 50, 51, 52, 53, 59, 60, 66]
# Average score = 82.981 % / Accuracy after training the model = 86.087 %
# fi = [39, 48, 49, 50, 51, 52, 59, 60, 66]

# Select only the desired features
X_selected = X[:, fi]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# # Standardize
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Train SVM and predict
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output predictions
print("Predictions:", y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Selecting features  ---------------------------------------------------------------------- Finish

# Clustering  ---------------------------------------------------------------------- Start

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

X_clustering = X

base_model = KMeans()
elbow = KElbowVisualizer(base_model, k=(1, 16), timings=False)
elbow.fit(X_clustering)
elbow.show(outpath="elbow_plot.png")

num_clusters= 3

# Fit KMeans
clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
clustering_model.fit(X_clustering)


# Plot clusters
def plot_pca_with_labels(data, labels, output_file_name):
    reducer = PCA(n_components=2)
    reduced_data = reducer.fit_transform(data)
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        plt.scatter(
            reduced_data[labels == label, 0],
            reduced_data[labels == label, 1],
            label=f'Cluster {label + 1}'
        )

    plt.title('KMeans Clustering (PCA-reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(output_file_name)

plot_pca_with_labels(data=X_clustering, labels=clustering_model.labels_, output_file_name="clusters.png")
plot_pca_with_labels(data=X_clustering, labels=y, output_file_name="clusters_y.png")

from sklearn.manifold import TSNE


def plot_tsne_with_labels(data, labels, output_file_name):
    # Apply t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    data_2d = tsne.fit_transform(data)

    # Plot the t-SNE projection with cluster colors
    plt.figure(figsize=(8, 6))
    for label in range(num_clusters):
        plt.scatter(
            data_2d[labels == label, 0],
            data_2d[labels == label, 1],
            label=f'Cluster {label + 1}',
            alpha=0.7
        )

    plt.title("KMeans Clustering Visualized with t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file_name)

plot_tsne_with_labels(data=X_clustering, labels=clustering_model.labels_, output_file_name="clusters_tsne.png")
plot_tsne_with_labels(data=X_clustering, labels=y, output_file_name="clusters_tsne_y.png")

# Clustering  ---------------------------------------------------------------------- Finish