I did a majority of the work over the course of my project on the Intel Open Vino Repo, and these are my final codes to get my desired output!

<img width="851" alt="Screenshot 2024-10-03 at 11 52 07 AM" src="https://github.com/user-attachments/assets/77ad1e44-8622-40e0-8af4-3b8c56a01f86">
<img width="1086" alt="Screenshot 2024-10-03 at 12 02 59 PM" src="https://github.com/user-attachments/assets/52b24821-e336-4a39-8db4-c7b7bf3af0e8">



DIATOS is a Diabetic Retinopathy diagnosis model and multimodal user interface that was custom trained from an initial 67% accuracy to utilizing the Intel AI PC's NPU preprocess and train a transfer learning model of Resnet50 finetuned on a kaggle Diabetic Retinopathy dataset. 

<img width="855" alt="Screenshot 2024-10-03 at 11 48 46 AM" src="https://github.com/user-attachments/assets/05551a04-f08b-4403-9671-48d43ee858b2">
<img width="853" alt="Screenshot 2024-10-03 at 11 50 28 AM" src="https://github.com/user-attachments/assets/ac144c16-e540-472f-bb5b-d2e00074bc3f">
<img width="850" alt="Screenshot 2024-10-03 at 11 50 36 AM" src="https://github.com/user-attachments/assets/bf0cbbdb-c043-4dd8-8885-c9f5a07bda8a">
<img width="850" alt="Screenshot 2024-10-03 at 11 50 46 AM" src="https://github.com/user-attachments/assets/13bf9b01-a117-45ca-84fc-b7ee44102d81">
<img width="1286" alt="Screenshot 2024-10-03 at 12 06 44 PM" src="https://github.com/user-attachments/assets/ddb4e2d5-bcd7-46ab-a1db-18756a7c746a">



6 different components of the OpenVINO ecosystem were utilized in the project. Qwen 2.5–0.6b-instruct was used as the natural language LLM, pulled from the OpenVINO repository toolkit, Intel's NPU Acceleration Library was utilized for preprocessing, Optimum was utilized for model export, TensorFlow and Pytorch OpenVINO integrations were utilized for the transfer learning.


<img width="850" alt="Screenshot 2024-10-03 at 11 50 58 AM" src="https://github.com/user-attachments/assets/a88ac362-76e0-4e10-847a-9a0a744d02d8">
<img width="851" alt="Screenshot 2024-10-03 at 11 51 07 AM" src="https://github.com/user-attachments/assets/b4a51bb9-a0da-4161-be78-7aed1b61498c">
<img width="854" alt="Screenshot 2024-10-03 at 11 51 17 AM" src="https://github.com/user-attachments/assets/a1fc52a6-6f37-4072-b067-d5e58cb58878">

<img width="1690" alt="Screenshot 2024-10-03 at 11 49 47 AM" src="https://github.com/user-attachments/assets/1232a5e0-d731-45a2-aba5-3a0f23fead53">


<img width="854" alt="Screenshot 2024-10-03 at 11 51 29 AM" src="https://github.com/user-attachments/assets/72133fb9-b665-49da-b77b-be2db6b6bfae">
<img width="849" alt="Screenshot 2024-10-03 at 11 51 39 AM" src="https://github.com/user-attachments/assets/432cb85a-196c-4c90-a556-80a02e2f115b">

The final result after boosting the model with Intel AI PC components was an 80% accuracy for categorizing 4 stages of Diabetic Retinopathy, yielding an impact of 30% increase in early stage Diabetic Retinopathy Detection, and 60% decrease in computational costs.


 <img width="852" alt="Screenshot 2024-10-03 at 11 51 50 AM" src="https://github.com/user-attachments/assets/da4a2d46-6201-4a35-9110-f1774d6ca0e1">
<img width="854" alt="Screenshot 2024-10-03 at 11 51 57 AM" src="https://github.com/user-attachments/assets/13dcce62-e121-4730-b96d-39b78875ab26">
