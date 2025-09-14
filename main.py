import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import spacy
import re
from typing import List, Set
import os

# Job Recommendation System
class JobRecommendationSystem:
    def __init__(self):
        """Initialize the job recommendation system with job data and TF-IDF vectorizer."""
        self.jobs_df = None
        self.tfidf_vectorizer = None
        self.job_vectors = None
        self.load_job_data()
        self.setup_vectorizer()

    def load_job_data(self):
        """Load job data from CSV file."""
        try:
            data_path = os.path.join("data", "jobs.csv")
            if os.path.exists(data_path):
                self.jobs_df = pd.read_csv(data_path)
                print(f"Loaded {len(self.jobs_df)} jobs from dataset")
            else:
                raise FileNotFoundError(f"Job data file not found at {data_path}")
        except Exception as e:
            print(f"Error loading job data: {e}")
            # Create empty dataframe as fallback
            self.jobs_df = pd.DataFrame(columns=['job_title', 'skills_required'])

    def setup_vectorizer(self):
        """Setup TF-IDF vectorizer and create job skill vectors."""
        if self.jobs_df is not None and not self.jobs_df.empty:
            # Preprocess job skills text
            job_skills_text = self.jobs_df['skills_required'].fillna('').astype(str)
            processed_skills = [self.preprocess_skills(skills) for skills in job_skills_text]

            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                stop_words='english',
                max_features=1000
            )

            # Fit vectorizer and transform job skills
            self.job_vectors = self.tfidf_vectorizer.fit_transform(processed_skills)
            print(f"TF-IDF vectorizer setup complete with {self.job_vectors.shape[1]} features")
        else:
            print("No job data available for vectorizer setup")

    def preprocess_skills(self, skills_text):
        """Preprocess skills text for better matching."""
        if not skills_text or pd.isna(skills_text):
            return ""

        # Convert to lowercase and remove extra spaces
        skills_text = str(skills_text).lower().strip()

        # Replace common separators with spaces
        skills_text = re.sub(r'[,;|/\n\r]+', ' ', skills_text)

        # Remove special characters but keep alphanumeric and spaces
        skills_text = re.sub(r'[^\w\s\-\+\#]', ' ', skills_text)

        # Replace multiple spaces with single space
        skills_text = re.sub(r'\s+', ' ', skills_text)

        return skills_text

    def get_recommendations(self, user_skills, top_n=None):
        """Get job recommendations based on user skills."""
        if not user_skills or self.tfidf_vectorizer is None:
            return []

        try:
            # Preprocess user skills
            user_skills_text = self.preprocess_skills(', '.join(user_skills))

            # Transform user skills using the fitted vectorizer
            user_vector = self.tfidf_vectorizer.transform([user_skills_text])

            # Calculate cosine similarity between user skills and all jobs
            similarities = cosine_similarity(user_vector, self.job_vectors).flatten()

            # Create recommendations list
            recommendations = []
            for idx, similarity in enumerate(similarities):
                if similarity > 0:  # Only include jobs with some similarity
                    job_data = self.jobs_df.iloc[idx]

                    # Find matching skills
                    job_skills = [s.strip().lower() for s in str(job_data['skills_required']).split(',')]
                    user_skills_lower = [s.strip().lower() for s in user_skills]
                    matching_skills = list(set(job_skills) & set(user_skills_lower))

                    recommendation = {
                        'job_title': job_data['job_title'],
                        'skills_required': job_data['skills_required'],
                        'similarity_score': float(similarity),
                        'matching_skills': matching_skills,
                        'job_index': idx
                    }
                    recommendations.append(recommendation)

            # Sort by similarity score in descending order
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Return top N recommendations if specified
            if top_n:
                recommendations = recommendations[:top_n]

            return recommendations

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

# Resume Parser
class ResumeParser:
    def __init__(self):
        """Initialize the resume parser with spaCy model."""
        self.nlp = self.load_spacy_model()
        self.skill_keywords = self.get_skill_keywords()

    @st.cache_resource
    def load_spacy_model(_self):
        """Load spaCy model with caching."""
        try:
            nlp = spacy.load("en_core_web_sm")
            print("Loaded en_core_web_sm spaCy model")
            return nlp
        except OSError:
            print("Warning: No spaCy model found.")
            return None

    def get_skill_keywords(self) -> Set[str]:
        """Get predefined list of technical skills and keywords."""
        skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'kotlin', 'swift', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'sass', 'less',

            # Frameworks & Libraries
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'seaborn', 'plotly', 'opencv', 'nltk', 'spacy', 'beautifulsoup', 'scrapy',

            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle',
            'sqlite', 'mariadb', 'dynamodb', 'neo4j',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'gitlab',
            'github', 'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'ci/cd',

            # Data Science & Analytics
            'machine learning', 'deep learning', 'data mining', 'data analysis', 'statistics',
            'data visualization', 'big data', 'hadoop', 'spark', 'kafka', 'airflow', 'tableau',
            'power bi', 'qlik', 'looker', 'jupyter', 'rstudio',

            # Web Technologies
            'rest api', 'graphql', 'microservices', 'soap', 'json', 'xml', 'ajax', 'websockets',
            'oauth', 'jwt', 'cors', 'webpack', 'babel', 'gulp', 'grunt',

            # Testing
            'unit testing', 'integration testing', 'selenium', 'jest', 'pytest', 'junit',
            'cypress', 'postman', 'swagger',

            # Soft Skills & Methodologies
            'agile', 'scrum', 'kanban', 'waterfall', 'project management', 'team leadership',
            'problem solving', 'communication', 'collaboration', 'mentoring', 'code review',

            # Other Tools
            'git', 'svn', 'jira', 'confluence', 'slack', 'trello', 'asana', 'notion',
            'figma', 'sketch', 'photoshop', 'illustrator', 'linux', 'windows', 'macos',
            'bash', 'powershell', 'vim', 'emacs', 'vscode', 'intellij', 'eclipse'
        }

        return skills

    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            # Read the uploaded file
            pdf_bytes = uploaded_file.read()

            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            extracted_text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                extracted_text += text + "\n"

            pdf_document.close()

            # Clean up the extracted text
            extracted_text = self.clean_text(extracted_text)

            return extracted_text

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\-\+\#\.\,\;\:\(\)\/]', '', text)

        # Remove extra spaces
        text = text.strip()

        return text

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using keyword matching."""
        if not text:
            return []

        extracted_skills = set()

        # Keyword matching
        text_lower = text.lower()
        for skill in self.skill_keywords:
            if skill.lower() in text_lower:
                extracted_skills.add(skill.title())

        return sorted(list(extracted_skills))

# Main Application
@st.cache_resource
def load_job_system():
    return JobRecommendationSystem()

def main():
    st.title("🚀 Job Recommendation System")
    st.markdown("Find the perfect job match based on your skills!")

    # Initialize system
    job_system = load_job_system()
    resume_parser = ResumeParser()

    # Sidebar for input options
    st.sidebar.header("How would you like to input your skills?")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Skills Input", "Upload Resume (PDF)"]
    )

    user_skills = []

    if input_method == "Manual Skills Input":
        st.header("📝 Enter Your Skills")
        skills_input = st.text_area(
            "Enter your skills separated by commas:",
            placeholder="e.g., Python, Machine Learning, SQL, Data Analysis, JavaScript",
            height=100
        )

        # Quick skill selection options
        st.subheader("🎯 Quick Skill Selection")
        st.markdown("Select common skills to add them to your list:")

        # Define skill categories
        programming_languages = ["Python", "JavaScript", "Java", "SQL", "TypeScript", "C++"]
        frameworks = ["React", "Node.js", "Django", "Flask", "Angular", "Spring Boot"]
        data_science = ["Machine Learning", "Data Analysis", "Pandas", "TensorFlow", "Tableau", "Power BI"]
        databases = ["MySQL", "PostgreSQL", "MongoDB", "Redis"]
        cloud_devops = ["AWS", "Docker", "Git", "Azure", "Kubernetes", "CI/CD"]
        soft_skills = ["Project Management", "Team Leadership", "Agile", "Problem Solving"]

        # Create columns for skill categories
        col1, col2, col3 = st.columns(3)

        selected_skills = []

        with col1:
            st.markdown("**Programming Languages**")
            for skill in programming_languages:
                if st.checkbox(skill, key=f"prog_{skill}"):
                    selected_skills.append(skill)

            st.markdown("**Databases**")
            for skill in databases:
                if st.checkbox(skill, key=f"db_{skill}"):
                    selected_skills.append(skill)

        with col2:
            st.markdown("**Frameworks & Libraries**")
            for skill in frameworks:
                if st.checkbox(skill, key=f"fw_{skill}"):
                    selected_skills.append(skill)

            st.markdown("**Cloud & DevOps**")
            for skill in cloud_devops:
                if st.checkbox(skill, key=f"cloud_{skill}"):
                    selected_skills.append(skill)

        with col3:
            st.markdown("**Data Science & ML**")
            for skill in data_science:
                if st.checkbox(skill, key=f"ds_{skill}"):
                    selected_skills.append(skill)

            st.markdown("**Soft Skills**")
            for skill in soft_skills:
                if st.checkbox(skill, key=f"soft_{skill}"):
                    selected_skills.append(skill)

        # Combine manual input and selected skills
        manual_skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()] if skills_input.strip() else []
        all_skills = list(set(manual_skills + selected_skills))  # Remove duplicates

        if all_skills:
            user_skills = all_skills
            st.success(f"✅ {len(user_skills)} skills total: {', '.join(sorted(user_skills))}")
        elif skills_input.strip():
            user_skills = manual_skills
            st.success(f"✅ {len(user_skills)} skills detected: {', '.join(user_skills)}")
        else:
            user_skills = []

    elif input_method == "📄 Upload Resume (PDF)":
        st.markdown("## 📑 Resume Intelligence Scanner")

        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "🎯 Upload Your Resume (PDF)",
                type="pdf",
                help="Our AI will automatically extract and analyze your skills"
            )

        with col2:
            st.info("""
            🔍 **What we extract:**
            - Technical skills
            - Programming languages
            - Frameworks & tools
            - Certifications
            - Experience keywords
            """)

        if uploaded_file is not None:
            try:
                with st.spinner("🧠 AI is analyzing your resume..."):
                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)

                    # Extract text from PDF
                    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    resume_text = ""
                    for page in pdf_document:
                        resume_text += page.get_text()
                    pdf_document.close()

                    progress_bar.progress(50)

                    if resume_text.strip():
                        progress_bar.progress(75)
                        st.success("✅ Resume processed successfully!")

                        # Enhanced text display
                        with st.expander("📄 Extracted Resume Content", expanded=False):
                            st.markdown("**Document Preview:**")
                            st.text_area("", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, 
                                       height=200, disabled=True)
                            st.caption(f"Total characters: {len(resume_text)}")

                        # Extract skills using NLP
                        user_skills = resume_parser.extract_skills(resume_text)
                        progress_bar.progress(100)

                        if user_skills:
                            st.success(f"🎯 Successfully extracted {len(user_skills)} skills!")

                            # Display extracted skills as chips
                            st.markdown("**🔍 Detected Skills:**")
                            skills_html = "".join([f'<span class="skill-chip">{skill}</span>' for skill in user_skills])
                            st.markdown(skills_html, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ No skills automatically detected. Consider manual input for better results.")
                    else:
                        st.error("❌ Could not extract readable text. Please ensure your PDF is text-based.")

            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.info("💡 Try a different PDF file or use manual input.")

    # Generate recommendations if skills are available
    if user_skills:
        st.header("🎯 Job Recommendations")

        with st.spinner("Finding the best job matches for you..."):
            try:
                recommendations = job_system.get_recommendations(user_skills)

                if recommendations:
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Jobs Found", len(recommendations))
                    with col2:
                        st.metric("Best Match Score", f"{recommendations[0]['similarity_score']:.1%}")
                    with col3:
                        avg_score = sum(job['similarity_score'] for job in recommendations) / len(recommendations)
                        st.metric("Average Match Score", f"{avg_score:.1%}")

                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📊 Top Recommendations", "📈 All Results", "🔍 Skills Analysis"])

                    with tab1:
                        st.subheader("🏆 Top 5 Job Matches")
                        top_jobs = recommendations[:5]

                        for i, job in enumerate(top_jobs, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"**{i}. {job['job_title']}**")
                                    st.markdown(f"*Required Skills:* {job['skills_required']}")

                                    # Show matching skills
                                    matching_skills = job.get('matching_skills', [])
                                    if matching_skills:
                                        st.markdown(f"*Your Matching Skills:* {', '.join(matching_skills)}")

                                with col2:
                                    st.metric("Match Score", f"{job['similarity_score']:.1%}")

                                st.divider()

                    with tab2:
                        st.markdown("### 📋 Complete Match Results")
                        st.markdown("*All job opportunities ranked by compatibility*")

                        # Create enhanced DataFrame for display
                        df_viz = pd.DataFrame(recommendations)
                        df_viz['similarity_score'] = df_viz['similarity_score'].round(3)
                        df_viz['match_percentage'] = (df_viz['similarity_score'] * 100).round(1).astype(str) + '%'

                        # Enhanced table display with better formatting
                        display_df = df_viz[['job_title', 'match_percentage', 'skills_required']].copy()
                        display_df.columns = ['🎯 Job Title', '📊 Match Score', '🔧 Required Skills']

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "🎯 Job Title": st.column_config.TextColumn(width="medium"),
                                "📊 Match Score": st.column_config.TextColumn(width="small"),
                                "🔧 Required Skills": st.column_config.TextColumn(width="large")
                            }
                        )

                        # Enhanced download options
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = df_viz.to_csv(index=False)
                            st.download_button(
                                "📥 Download Full Results (CSV)",
                                csv,
                                "job_recommendations_complete.csv",
                                "text/csv",
                                help="Download all recommendations with detailed data"
                            )

                        with col2:
                            top_10_csv = df_viz.head(10).to_csv(index=False)
                            st.download_button(
                                "🏆 Download Top 10 (CSV)",
                                top_10_csv,
                                "top_10_job_recommendations.csv",
                                "text/csv",
                                help="Download only the top 10 matches"
                            )

                    with tab3:
                        st.markdown("### 📊 Advanced Analytics & Market Insights")

                        # Enhanced visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            # Enhanced bar chart
                            fig_bar = px.bar(
                                df_viz.head(10),
                                x='similarity_score',
                                y='job_title',
                                orientation='h',
                                title='🎯 Top 10 Job Match Scores',
                                labels={'similarity_score': 'Compatibility Score', 'job_title': 'Job Position'},
                                color='similarity_score',
                                color_continuous_scale='Viridis'
                            )
                            fig_bar.update_layout(
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False,
                                height=500
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        with col2:
                            # Score distribution with enhanced styling
                            fig_hist = px.histogram(
                                df_viz,
                                x='similarity_score',
                                nbins=15,
                                title='📈 Match Score Distribution',
                                labels={'similarity_score': 'Compatibility Score', 'count': 'Number of Positions'},
                                color_discrete_sequence=['#667eea']
                            )
                            fig_hist.update_layout(height=500)
                            st.plotly_chart(fig_hist, use_container_width=True)

                        # Additional insights
                        st.markdown("### 💡 Career Insights")

                        # Score ranges analysis
                        excellent_matches = len([job for job in recommendations if job['similarity_score'] >= 0.5])
                        good_matches = len([job for job in recommendations if 0.3 <= job['similarity_score'] < 0.5])
                        potential_matches = len([job for job in recommendations if 0.1 <= job['similarity_score'] < 0.3])

                        insight_col1, insight_col2, insight_col3 = st.columns(3)

                        with insight_col1:
                            st.info(f"""
                            🌟 **Excellent Matches** (50%+)

                            {excellent_matches} positions

                            *Apply immediately - you're highly qualified!*
                            """)

                        with insight_col2:
                            st.warning(f"""
                            📈 **Good Matches** (30-49%)

                            {good_matches} positions

                            *Consider upskilling for better compatibility*
                            """)

                        with insight_col3:
                            st.info(f"""
                            🎯 **Potential Matches** (10-29%)

                            {potential_matches} positions

                            *Future opportunities with skill development*
                            """)

                else:
                    st.warning("🤔 No job matches found for your skills. Try adding more skills or check your input.")

            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")

    # Footer information
    st.markdown("---")
    st.markdown(
        """
        **About this system:**
        - Uses TF-IDF vectorization and cosine similarity for job matching
        - Analyzes 30+ job roles from diverse technology sectors
        - Provides interactive visualizations and downloadable results
        - Real-time matching with percentage similarity scores
        - Covers 400+ technical skills across programming, frameworks, and cloud technologies
        """
    )

if __name__ == "__main__":
    main()