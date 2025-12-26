Project Setup & Usage Guide
1. Overview

    This project uses Streamlit for an interactive user interface and integrates multiple tools for data handling, embeddings, visualization, and AI model responses.


    Key functionalities include:

    Visualizing and analyzing datasets (using Pandas, Matplotlib, Seaborn)
    Generating embeddings (using SentenceTransformers and ChromaDB)
    Integrating with Google Generative AI and Groq LLM
    Securely managing API keys via .env

2. Installation
    Step 1: Clone or Download the Project

        If your project is in a Git repository:

        git clone <repository_url>
        cd <project_folder>


        Or, if you have the files manually, just open the folder in your terminal.


    Step 2: Set Up a Virtual Environment (Recommended)

        For Windows:

        python -m venv venv
        venv\Scripts\activate


        For macOS/Linux:

        python3 -m venv venv
        source venv/bin/activate

    Step 3: Install Required Dependencies

        Make sure your requirements.txt file is in the project root.

        pip install -r requirements.txt


        This installs:

        streamlit → UI framework

        numpy, pandas → data manipulation

        seaborn, matplotlib → visualization

        chromadb, sentence-transformers → vector storage and embeddings

        google-generativeai, groq → AI model access

        python-dotenv → for loading API keys


3. Environment Setup

    Create a .env file in your project root directory and add your API keys:

    GOOGLE_API_KEY=your_google_genai_key
    GROQ_API_KEY=your_groq_key

    (Replace with your actual API keys — never commit this file to Git.)


4. Running the Application

    To start your Streamlit app:

    streamlit run app.py


    This will open a local server, usually at:

    http://localhost:8501


    Open it in your browser to interact with the UI.

5. File Descriptions
    File / Folder	Description
    app.py	Main Streamlit application. Handles UI, logic, embeddings, and API calls.
    requirements.txt	Contains all necessary Python libraries for the project.
    .env	Stores sensitive API keys for security.
    CO2_Emission_Dataset_200.csv	Dataset used for analysis and visualization.

6. Common Issues & Fixes
    Issue	Cause	Fix
    ModuleNotFoundError	Missing dependencies	Run pip install -r requirements.txt
    .env not loaded	Wrong file location or name	Ensure .env is in the same directory as app.py
    Streamlit not launching	Port already in use	Run with streamlit run app.py --server.port 8502
    API key errors	Invalid or missing keys	Check your .env values and re-run

7. Optional Enhancements

    Add logging with logging module for better debugging.

    Use Streamlit secrets for deployment security.

    Cache embeddings using st.cache_data for faster reloads.

    Add a requirements lock file to ensure dependency consistency.

8. Example Command Summary
    # Create virtual environment
    python -m venv venv

    # Activate it
    source venv/bin/activate     # Linux/Mac
    venv\Scripts\activate        # Windows

    # Install dependencies
    pip install -r requirements.txt

    # Run the app
    streamlit run app.py