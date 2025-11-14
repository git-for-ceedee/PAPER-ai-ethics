AIML399 Pick-a-path Ethics&Results (PAPER) Project Readme
1.1	Overview
An interactive, choose-your-own-adventure style training application for exploring AI ethics scenarios in New Zealand workplaces. Built with Retrieval-Augmented Generation (RAG) technology, this application generates contextual, relevant AI ethics scenarios based on user selections and retrieves supporting evidence from a curated knowledge base.
PAPE is designed to help professionals understand and navigate AI ethics challenges through interactive storytelling. Users select their industry, role, and ethical topic of interest, and the system generates personalised scenarios with multiple choice paths. Each choice leads to consequences and outcomes, creating an engaging learning experience grounded in real-world AI ethics principles.
1.2	Running the PAPER app	lication
The PAPER application is an online AI ethics teaching tool. There are 2 parts to it – the application itself and an analytics dashboard. It is live and hosted in the cloud via Streamlit.io. Simply click on the links below to access: 
•	The PAPER application - https://paper-ai-ethics.streamlit.app/. 
•	The analytics dashboard - https://paper-ai-ethics-analytics.streamlit.app/ .
If PAPER has opened correctly you will see the welcome page:
There are various settings you can try out which will give you different information:
•	Query settings – which industry, role, ethical topic you’d like to explore
•	Response settings - # data reference chunks to retrieve, creativity of answers, story size
•	Feedback button – click here to give feedback on the tool
•	LLM settings – show/hide full prompt, chat-gpt model to use, API usage

The analytics dashboard will show options on the left and several graphs. It includes: 
•	Theme (topic) distribution – shows the most popular stories (defaults to ‘Random ethics topic’)
•	Latency over time(ms) – monitoring technical performance of chat-GPT API
•	Reading ease – If scores drop too low, the AI is generating overly complex text that may alienate learners. If too high, content might seem oversimplified. The ideal range (60-80) balances accessibility with sophistication.
•	Repeat similarity – how different are the stories (given the same input from the RAG database). High similarity means the AI is essentially repeating itself, which provides poor learning value. Users want fresh, diverse scenarios. This metric directly measures content quality and variety.
•	Event type – stories started/completed/generation failures. Describes how well the app is functioning, drop-off points and completion rates. Good for monitoring technical performance.
•	Deepest depth reached – how far did the person go into the story (max. 3 levels). Gives a rough indication of how deeply users engage with content. In the absence of direct observation, this is a proxy for user engagement.
Should you wish for further information, this link will give you the full repository, including a copy of the code, the database and raw data files. This is the repository on which the live system is operating.
Git repository: https://github.com/git-for-ceedee/PAPER-ai-ethics 
1.3	File structure
Pick_a_path_AI_ethics/
├── src/
│   ├── PAPEui.py                    		# Main Streamlit UI application
│   ├── PAPEsetup_database.py       	# Database schema initialization
│   ├── PAPEload_structured_data.py 	# Load CSV data into database
│   ├── PAPEdata_load.py            		# Ingest documents and build FAISS index
│   ├── PAPEingest_web_sources.py   	# Web scraping utilities
│   ├── PAPEtag_themes.py           		# Theme tagging for chunks
│   ├── PAPEsmoke_test.py           		# RAG query testing
│   ├── PAPEanalysis.py             		# Analytics dashboard
│   ├── PAPEsecurity.py             		# Security validation functions
│   ├── PAPErate_limiter.py         		# API rate limiting
│   └── PAPElogging_config.py       		# Centralized logging configuration
├── data/
│   ├── datafiles/                  		# CSV files with structured data
│   ├── raw_cases/                  		# Source documents (PDF, DOCX)
│   └── archive/                    		# Archived data files
├── vectorstore/                    		# FAISS index and chunk IDs
│   ├── chunks.faiss               		# FAISS vector index
│   └── chunk_ids.csv              		# Mapping of index to chunk IDs
├── ragEthics.db                    		# SQLite database
├── requirements.txt                		# Python dependencies
└── README.doc                       		# This file
1.4	Building from scratch with OpenAI chat-gpt API integration
1.4.1	Prerequisites
You will need:
•	Python 3.8 or higher
•	OpenAI API key
•	Google Service Account credentials (for logging to Google Sheets - optional)
•	Virtual environment (recommended)
•	A copy of https://github.com/git-for-ceedee/PAPER-ai-ethics 
Check these files are included in the repository:
•	Requirements.txt
•	.gitignore
The following libraries must be installed in order to run the application. You can check them by running CMD> pip list in the working directory you have created:
•	CountVectorizer
•	faiss
•	numpy
•	pandas
•	python-docx
•	pymupdf
•	sentence-transformers
•	streamlit 
•	textstat
•	trafilatura
1.4.2	Installation
Create a working directory. Download the git repository into your working directory. Start a powershell session.
In your working terminal/powershell session:
	Cd c:\Your_working_directory
	pip install --upgrade pip
	pip install -r requirements.txt
	.\venv_rag\Scripts\activate
o	once active your prompt should show (venv_rag) at the beginning
	python -V
o	returns the version of python installed
o	If it is not listed then try:
o	python -c "import sys; print(sys.executable)”
•	pip list | findstr /I "CountVectorizer faiss numpy pandas pymupdf sentence-transformers streamlit textstat trafilatura "
o	You should see these packages listed. If not please install them.
Database Setup & test
	python -m src.PAPEsetup_database
	python -m src.PAPEdata_load
	python -m src.PAPEload_structured_data_simple
	dir .\ragEthics.db
o	you should see where the db has been created
	dir .\vectorstore
o	you should see the location of the db vector files
	python -m src.PAPEsmoke_test
o	testing retrieval of chunks without LLM
o	you will be asked how many chunks you want to retrieve
Check LLM interface is working
You will need a valid OpenAI API key (or other LLM of your choice), a Google service account and to set up a google sheet which allows access to write to the spreadsheet. 
Create a file: .streamlit/secrets/toml . In that file create these key environment variables:
	OPENAI_API_KEY = "sk-..."
GSHEET_SPREADSHEET_NAME = "PAPE User Metrics"
GSHEET_WORKSHEET_NAME = "logs"
GOOGLE_SERVICE_ACCOUNT_JSON = """{ ... }"""
If OpenAI connection fails:
-	Check your API key in `.streamlit/secrets.toml`
-	Verify the key is valid and has credits
-	Check your internet connection
Startup the App
	streamlit run src\PAPEui.py
o	This starts up the application
	streamlit run src\PAPEanalysis.py
o	This starts the analytics dashboard
Closing the app
	Close the app window
	Return to Terminal B and press CTRL+C to close the streamlit session.
This process checks:
•	Python is working
•	The correct packages are installed
•	The db exists
•	There is data in the db
•	The LLM is installed & retrieving information
•	Runs the front end
•	Runs the dashboard
1.5	Data and logging
Each user session generates rows in the Google Sheet (with csv fallback) including:
timestamp, session_id, event_type, story_key, user_industry, user_role, user_topic, user_query, level, latency_ms, flesch_ease, fk_grade, repeat_sim, choices_count, details
These metrics support charts on levels reached, reading ease (Flesch Reading Ease), text similarity, and story depth, available via the dashboard.
If Google Sheets logging fails:
•	Verify service account JSON in secrets
•	Check spreadsheet permissions
•	Application will fall back to CSV logging automatically
1.6	Acknowledgements
This is an academic project for AIML339. For questions or contributions, please contact the project maintainer. Appreciation for the contributions from:
•	Centre for Data Ethics and Innovation, Statistics NZ
•	OpenAI for LLM API access
•	Streamlit for the web framework
Note: This application is designed for educational and training purposes. Always consult with ethics experts and legal advisors for real-world AI ethics decisions.
There are no known bugs in the executables. This is all my own work with tutoring assistance from chat-GPT and Claude. Security checks & updates by Claude Code.
1.6.1	License
The programme is available for anyone to use – CC BY-NC-SA licence.
