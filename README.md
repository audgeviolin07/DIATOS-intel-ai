I did a majority of the work over the course of my project on the Intel Open Vino Repo, and these are my final codes to get my desired output!

DIATOS is a Diabetic Retinopathy diagnosis model and multimodal user interface that was custom trained from an initial 67% accuracy to utilizing the Intel AI PC's NPU preprocess and train a transfer learning model of Resnet50 finetuned on a kaggle Diabetic Retinopathy dataset. 6 different components of the OpenVINO ecosystem were utilized in the project. Qwen 2.5–0.6b-instruct was used as the natural language LLM, pulled from the OpenVINO repository toolkit, Intel's NPU Acceleration Library was utilized for preprocessing, Optimum was utilized for model export, TensorFlow and Pytorch OpenVINO integrations were utilized for the transfer learning. The final result after boosting the model with Intel AI PC components was an 80% accuracy for categorizing 4 stages of Diabetic Retinopathy, yielding an impact of 30% increase in early stage Diabetic Retinopathy Detection, and 60% decrease in computational costs.

<img width="855" alt="Screenshot 2024-10-03 at 11 48 46 AM" src="https://github.com/user-attachments/assets/05551a04-f08b-4403-9671-48d43ee858b2">
