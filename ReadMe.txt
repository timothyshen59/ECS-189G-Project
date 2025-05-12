The project trains and evaluates Convoluted Neural Networks (CNN) on different image classification datasets (ORL, MNIST, CIFAR)

How To Run
1) Navigate to the script_cnn.py file in PyCharm
    - location: code/script/stage_3_script/script_cnn.py
2) Right click on the file and click "Run 'script_cnn'"
    - This will call classes and functions within code/code/stage_3_code/ to train and evaluate the model
3) To change the dataset the CNN will train on, open script_cnn.py and modify data_obj.dataset_file_name string variable to dataset of choice (on line 20)
    - Possible Datasets are: 'ORL', 'CIFAR', or 'MNIST'
    - The functions/classes will dynamically adjust to the dataset


Outputs
    - Test Accuracy
    - For each epoch, output Accuracy, F1-Score, Precision, Recall, Loss
    - Plots on Training Loss Curve/Accuracy/F1-Score/Precision/Recall over epochs for most recent runs 
    - Saved under code/result/stage_3_result as a png

The dataset files are NOT included in the GitHub repository due to GitHub's 100 MB file size limit.
    - You will need to download the data (ORL/CIFAR/MNIST) and place them under code/data/stage_3_data/