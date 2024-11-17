AI-Powered Climate Change Tracker 🌍
Welcome to the AI-Powered Climate Change Tracker GitHub repository! This project was submitted for the Vultr Hackathon and showcases an innovative approach to predicting climate trends and weather conditions using AI.

The project leverages Streamlit for the frontend interface and integrates a Laravel-based chatbot powered by Vultr services. Due to time constraints, the entire project couldn't be developed in Laravel, but the chatbot functionality has been seamlessly integrated into the Streamlit application.

🚀 Live Demo
Check out the live application here: AI-Powered Climate Change Tracker (https://weatherpre.streamlit.app/)

📂 Repository Structure
The repository consists of the following files and folders:

Streamli_C.py:
The main running file for the project, which uses Streamlit to display the output of our climate predictions and weather analysis. This file integrates the Laravel-built chatbot for enhanced functionality.

Laravel Files:
Contains the chatbot code developed using Laravel and PHP. While we initially planned to build the entire project in Laravel, we focused on delivering a robust integration within Streamlit due to time constraints.

Other Supporting Files:
Includes files for styling, datasets, and utility scripts required for the project.

🔧 Technologies Used
Frontend:
Streamlit
Laravel (for chatbot development)
Backend:
Python (for AI/ML-based climate predictions)
PHP (for the chatbot functionality using Laravel)
Cloud Infrastructure:
Vultr API for serverless inference

💡 Features
Climate Prediction:

Predict climate trends for different timelines (1 month, 6 months, 1 year, and 2 years) using AI models.
Weather Analysis:

Upload an image to predict weather conditions based on advanced deep learning models.
Integrated Chatbot:

A Laravel-based chatbot powered by Vultr services for interactive assistance.
User-Friendly Interface:

Built with Streamlit for a seamless and intuitive experience.
🛠️ How to Run Locally
Prerequisites
Python 3.x
Streamlit
Steps
Clone the repository:

git clone https://github.com/Tanny28/Weather_predict.git
cd Weather_predict

Install Python dependencies:
pip install -r requirements.txt

Run the Streamlit application:

streamlit run Streamli_C.py 
📢 Notes
The main application is powered by Streamli_C.py.
The Laravel files are part of the chatbot integration and can function independently for further development.
📌 Future Improvements
Complete migration of the project to Laravel for a unified framework.
Enhance the chatbot's capabilities and features.
Expand predictive models for more comprehensive climate and weather analysis.
🙌 Acknowledgments
This project was developed by Team Techovators as a submission for the Vultr Hackathon. A special thanks to Vultr for their powerful cloud infrastructure and the hackathon organizers for the opportunity to innovate.

Feel free to explore, fork, and contribute!


