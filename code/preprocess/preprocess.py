from data_cleaning import data_cleaning_run
from data_cleaning_num import data_cleaning_num_run
from data_cleaning_ch import data_cleaning_ch_run
from data_combine import data_combine
from denoising import denoising

if __name__ == '__main__':
    data_cleaning_run()
    data_cleaning_num_run()
    data_cleaning_ch_run()
    data_combine()
    denoising()
    print("预处理完成")
