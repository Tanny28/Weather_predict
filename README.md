# 🌟 AI-Powered Climate Change Tracker 🌍  
> **A Vultr Hackathon Submission**

Welcome to the **AI-Powered Climate Change Tracker** repository!  
This project combines **AI**, **Streamlit**, and **Laravel** to provide predictive climate insights and interactive weather analysis.  
The project leverages Streamlit for the frontend interface and integrates a Laravel-based chatbot powered by Vultr services. Due to time constraints, the entire project couldn't be developed in Laravel, but the chatbot functionality has been seamlessly integrated into the Streamlit application.


---

## 🚀 **Live Demo**  
👉 Check out the live application here:  
[![AI-Powered Climate Change Tracker](https://img.shields.io/badge/Live_App-Click_Here-brightgreen?style=for-the-badge&logo=streamlit)](https://weatherpre.streamlit.app/)  

---

## 📂 **Repository Structure**  
Here’s a quick overview of the files and their roles:  

```plaintext
📦 Weather_predict
├── Streamli_C (web_app.py)  # Main running file (Streamlit-based web application)
├── Laravel_Chatbot/         # Contains the Laravel PHP files for the chatbot
├── requirements.txt         # Python dependencies for Streamlit app
└── Other_Files/             # Supporting files (datasets, utility scripts, etc.)
```
Streamli_C(web_app.py): The core application that runs the Streamlit web app and integrates the Laravel chatbot.
Laravel_Chatbot/: Chatbot functionality built in Laravel. Due to time constraints, this part is integrated into the Streamlit app instead of a full Laravel implementation.

## 💻 Technologies Used
Technology	Purpose
Streamlit	Frontend web interface
Laravel PHP	Chatbot development
Python	AI and ML-based climate predictions
Vultr API	Serverless inference integration

## 🎯 Features
✨ Climate Predictions: Predict trends for timelines like 1 month, 6 months, 1 year, and 2 years.
🌦️ Weather Analysis: Upload images to get weather predictions using deep learning models.
🤖 Integrated Chatbot: A Laravel-based chatbot for interactive assistance, powered by Vultr.
🖥️ User-Friendly Interface: Built with Streamlit for an intuitive experience.

## 🛠️ Setup Instructions

Prerequisites
Ensure you have the following installed:

Python 3.x
Streamlit
Laravel PHP Framework

Steps to Run Locally
Clone the repository:
git clone https://github.com/Tanny28/Weather_predict.git
cd Streamlit_C

Install Python dependencies:

pip install -r requirements.txt

Run the Streamlit application:
streamlit run web_app.py (from Streamlit_C)

📌 Project Notes
The main running file is Streamli_C (web_app.py).
Laravel files are specific to the chatbot and can be extended for future improvements.
🛠️ Future Plans
🚀 Fully migrate the project to Laravel for a unified framework.
🤖 Enhance chatbot capabilities and interactions.
🌐 Expand predictive models for global insights.

🙌 Acknowledgments
We extend our gratitude to Vultr for their powerful cloud services and the Vultr Hackathon team for providing this opportunity.



