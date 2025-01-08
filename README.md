# Paper Similarity Checker

This project is a web-based tool for analyzing and visualizing the similarity of submissions across different tracks, specifically designed for PCS (Paper Submission and Review Systems). It uses the **TF-IDF (Term Frequency-Inverse Document Frequency)** method combined with **cosine similarity** to compute and compare textual similarities between abstracts or titles of submissions.

## Key Features:
- **Compare Across Tracks**: Upload two datasets of submissions from different tracks and compute their similarity.
- **Single Dataset Analysis**: Analyze similarity within a single dataset of submissions.
- **Interactive Visualization**: 
  - View the similarity matrix with a customizable gradient threshold.
  - Automatically highlights values below the threshold in gray for better clarity.
- **Top Matches**: Displays the top 10 most similar papers for easy identification of potential duplicates or related submissions.
- **Configurable Threshold**: Allows the user to adjust the similarity threshold dynamically.

## How It Works:
1. Upload submission data as Excel files.
2. Ensure that the dataset includes either an `Abstract` or `Title` column and optionally a `Decision` column for filtering.
3. The tool computes similarities based on the chosen text columns and visualizes results interactively in a user-friendly table.

## Use Case:
The tool is ideal for conference organizers and reviewers to detect:
- Overlapping or duplicate submissions across tracks.
- Highly similar papers within a single track.

## Requirements:
- Python 3.8+
- Dependencies: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`

## Installation and Usage:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paper-similarity-checker.git
   
2. Navigate to the project directory:
    ``` cd paper-similarity-checker

3. Install dependencies:
    ```pip install -r requirements.txt

4. Run the Streamlit app:
    ``` streamlit run app.py
    
Upload your datasets and analyze the similarities.