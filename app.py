import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Function to compute similarity between abstracts in two dataframes
# If 'Abstract' column is missing, use 'Title' column instead
def compute_similarity_between_dataframes(df1, df2):
    # Set Paper ID as index if exists
    if 'Paper ID' in df1.columns:
        df1 = df1.set_index('Paper ID')
    if 'Paper ID' in df2.columns:
        df2 = df2.set_index('Paper ID')

    # Determine which column to use
    df1_column = 'Abstract' if 'Abstract' in df1.columns else 'Title'
    df2_column = 'Abstract' if 'Abstract' in df2.columns else 'Title'

    # Extract the relevant columns
    df1_texts = df1[df1_column].dropna()
    df2_texts = df2[df2_column].dropna()

    # Combine texts from both dataframes
    combined_texts = pd.concat([df1_texts, df2_texts])

    # Compute TF-IDF vectors for all texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(df1_texts)], tfidf_matrix[len(df1_texts):])

    # Return the similarity matrix
    return pd.DataFrame(similarity_matrix, index=df1_texts.index, columns=df2_texts.index)

# Function to compute similarity within a single dataframe
# If 'Abstract' column is missing, use 'Title' column instead
def compute_similarity_within_dataframe(df):
    # Set Paper ID as index if exists
    if 'Paper ID' in df.columns and df.index.name != 'Paper ID':
        df = df.set_index('Paper ID', drop=False)

    # Determine which column to use
    column = 'Abstract' if 'Abstract' in df.columns else 'Title'

    # Extract the relevant column
    texts = df[column].dropna()

    # Compute TF-IDF vectors for the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Return the similarity matrix
    return pd.DataFrame(similarity_matrix, index=texts.index, columns=texts.index)

# Function to display similarity matrix with gradient
def display_similarity_with_gradient(similarity_matrix, threshold):
    # Convert similarity to percentage
    similarity_percentage = similarity_matrix * 100

    # Apply a gradient with threshold filtering
    styled_df = similarity_percentage.style.background_gradient(cmap="coolwarm")
    styled_df = styled_df.map(lambda v: "background-color: lightgray;" if v < threshold else "")
    styled_df = styled_df.format("{:.2f}%")
    st.write(styled_df)

# Function to display top 10 most similar papers
def display_top_similar_pairs(similarity_matrix, df1, df2):
    # Flatten the similarity matrix and get the indices
    similarity_flat = similarity_matrix.stack().reset_index()
    similarity_flat.columns = ['Paper1', 'Paper2', 'Similarity']

     # Filter out self-similarities (only for within-dataset analysis)
    if df1.equals(df2):
        similarity_flat = similarity_flat[similarity_flat['Paper1'] != similarity_flat['Paper2']]

    top_similarities = similarity_flat.sort_values(by='Similarity', ascending=False).head(10)

    # Display the results
    st.write("### Top 10 Most Similar Papers")
    for _, row in top_similarities.iterrows():
        paper1 = df1.loc[row['Paper1']]
        paper2 = df2.loc[row['Paper2']]
        st.write(f"**Paper 1 ID:** {row['Paper1']} | **Title:** {paper1['Title']} | **Authors:** {paper1['Authors']}")
        st.write(f"**Paper 2 ID:** {row['Paper2']} | **Title:** {paper2['Title']} | **Authors:** {paper2['Authors']}")
        st.write(f"**Similarity Score:** {row['Similarity']:.2f}%")
        st.markdown("---")

# Streamlit app
def main():
    st.title("Paper Similarity Checker")

    st.sidebar.title("Upload Data")

    uploaded_file1 = st.sidebar.file_uploader("Upload first Excel file", type="xlsx")
    uploaded_file2 = st.sidebar.file_uploader("Upload second Excel file", type="xlsx")

    threshold = st.sidebar.slider("Similarity Threshold (%)", min_value=0, max_value=100, value=50, step=1)

    if uploaded_file1 is not None and uploaded_file2 is not None:
        df1 = pd.read_excel(uploaded_file1)
        df2 = pd.read_excel(uploaded_file2)

        # Filter dataframes by 'Decision' column
        df1 = df1[df1['Decision'] == 'A'] if 'Decision' in df1.columns else df1
        df2 = df2[df2['Decision'] == 'A'] if 'Decision' in df2.columns else df2

        st.write("### Dataframe 1")
        st.write(df1.head())

        st.write("### Dataframe 2")
        st.write(df2.head())

        similarity_matrix = compute_similarity_between_dataframes(df1, df2)

        st.write("### Similarity Matrix with Gradient")
        display_similarity_with_gradient(similarity_matrix, threshold)

        #display_top_similar_pairs(similarity_matrix, df1, df2)

    elif uploaded_file1 is not None:
        df1 = pd.read_excel(uploaded_file1)

        # Filter dataframe by 'Decision' column
        df1 = df1[df1['Decision'] == 'A'] if 'Decision' in df1.columns else df1

        st.write("### Dataframe")
        st.write(df1.head())

        similarity_matrix = compute_similarity_within_dataframe(df1)

        st.write("### Similarity Matrix with Gradient")
        display_similarity_with_gradient(similarity_matrix, threshold)

        #display_top_similar_pairs(similarity_matrix, df1, df1)

    else:
        st.write("Please upload at least one Excel file to proceed.")

if __name__ == "__main__":
    main()