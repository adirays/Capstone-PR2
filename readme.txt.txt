HEART DISEASE PREDICTION SYSTEM
(Model Training + Streamlit Frontend)

--------------------------------------------------

REQUIREMENTS
- Python 3.8 or above
- pip installed

--------------------------------------------------

STEP 1: OPEN PROJECT FOLDER

Open terminal / command prompt inside the project folder.

--------------------------------------------------

STEP 2: CREATE & ACTIVATE VIRTUAL ENVIRONMENT

Create virtual environment (only once):
python -m venv .venv

Activate virtual environment:

Windows:
.venv\Scripts\activate

macOS / Linux:
source .venv/bin/activate

--------------------------------------------------

STEP 3: INSTALL REQUIRED PACKAGES

Install all dependencies using requirements.txt:

pip install -r requirements.txt

--------------------------------------------------

STEP 4: TRAIN THE MODEL

Run the model training script:

python train_model.py

This will:
- Train the Random Forest model
- Evaluate the model
- Save the trained model as heart_model.pkl

--------------------------------------------------

STEP 5: RUN THE FRONTEND

Start the Streamlit application:

streamlit run app.py

A browser window will open automatically.
If not, open the localhost URL shown in the terminal.

--------------------------------------------------

IMPORTANT NOTES

- Always activate the virtual environment before running commands
- Run train_model.py only once
- Do not delete heart_model.pkl
- app.py can be run multiple times

--------------------------------------------------
