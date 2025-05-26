from local_code.stage_4_code.Dataset_Loader_Gen import Dataset_Loader_Gen
from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_4_code.Setting_Train_Test_Split_Gen import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Joke_Method_RNN import Method_RNN
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------


    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader_Gen('stage_4_data', '')
    data_obj.load()
    data_obj.save_vocab('../../data/stage_4_data/text_generation/vocab_dict.pth')
    print("Saved .pth file successfully")

    method_obj = Method_RNN('RNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/'
    result_obj.result_destination_file_name = 'prediction_result_joke'

    setting_obj = Setting_Train_Test_Split('', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()

    # ------------------------------------------------------