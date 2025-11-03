# Job-Recommendation-System

check website here https://job-recommendation-jgyajfhrnhj3lzjbmxs5qn.streamlit.app/

A lightweight Streamlit app that recommends jobs based on your skills using TF‑IDF vectorization and cosine similarity.

## Features
- Upload a PDF resume or enter skills manually.
- Extracts skills from resumes using keyword matching (optional spaCy support).
- Matches user skills against roles in `data/jobs.csv`.
- Interactive Streamlit UI with charts and CSV export.

## Quickstart
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the app:
   ```sh
   streamlit run app.py
   ```
3. Go to `http://localhost:8501` in your browser.

## Usage
- Upload your resume or input your skills.
- Select the job title you are interested in.
- View the recommended jobs and their details.
- Download the recommendations as a CSV file.

## Technologies
- Python
- Streamlit
- Pandas
- Scikit-learn
- NLTK
- SpaCy (optional)

## Data
- Job roles and requirements are sourced from `data/jobs.csv`.
- Resume parsing uses `nltk` and `spacy` for natural language processing.

## Contributing
1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes
4. Commit your changes: `git commit -m 'Add new feature'`
5. Push to the branch: `git push origin feature-branch`
6. Submit a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Inspired by real-world job recommendation systems.
- Built as a capstone project for the Data Science program at XYZ University.

## Contact
- Your Name - [Shishank](mailto:shishank1505@gmail.com)
- LinkedIn: [shishank1](https://www.linkedin.com/in//shishank1/)
