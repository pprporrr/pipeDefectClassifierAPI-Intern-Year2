# pipeDefectClassifierAPI-Intern-Year2
This repository contains the implementation of an artificial intelligence project conducted in collaboration with the Electricity Generating Authority of Thailand (EGAT) and the Research and Consulting Institute of Thammasat University. The project aims to leverage artificial intelligence technology for the analysis of causes of boiler damage. As part of this project, a website has been developed to allow users to upload images and obtain predicted results. The primary focus of my contribution was on the machine learning component of the project.
The goal of this project is to apply computer vision techniques to analyze boiler damage causes. 

## Getting Started
To use the provided API for boiler damage classification, follow these steps:
 1. Clone this repository to your local machine.
 2. Navigate to the project directory.
### Prerequisites
- Docker
- Docker Compose
### Running the API
In your terminal, run the following command to build and run the API:
- docker-compose up --build -d

### The API will be accessible at the following endpoints:
- Xception model (TensorFlow Serving): http://localhost:8501/
- FastAPI classifier: http://localhost:8000/
  
## Conclusion
This repository contains the code and documentation for the implementation of an AI-powered boiler damage analysis project. The provided API allows users to upload images and receive predictions based on the trained model. The process involves data collection, preprocessing, model selection, training, evaluation, optimization, and testing. The collaborative effort between EGAT, Thammasat University, and my contribution has resulted in a functional AI solution for analyzing boiler damage causes.
