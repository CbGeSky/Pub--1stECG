The sample codes contains the following components:

* Python scripts:
   -- challenge.py (necessary) -  add your codes to classify normal and diseases.For ease of evaluation, you should pay attention to the following points:
   1.You need to write the results into "answers.csv",and save it in the current folder
   2.You need to write your test data path with the argparse parameter
   In short, challenge.py is your test code to make predictions or inferences. Please refer to this demo file for details.


* BASH scripts:
   -- run.sh (necessary) - a script calls "challenge.py" to generate "answers.csv", you can modify the --test_path parameter in this file
     
	 
* CSV files:
   -- answers.csv (necessary) - a text file containing the prediction results.

* README.txt - this file

* Other files:
     These files support to run the bash file and the challenge.py, such as your codes to run the model, and the model file, etc.

We verify that your code is working as you intended, by running "run.sh" on the test set, then comparing the results with references.

## 说明
    文件组成情况
    |__ answers.csv     由测试集数据预测生成的标签文件
    |__ chanllenge.py   主体代码，完成测试集数据的读取，模型的读取，并给出预测结果
    |__ f_model.py      模型的定义
    |__ f_preprocess.py 包含一些预处理的过程
    |__ README.txt      简单的说明，未编辑
    |__ run.sh          shell下的执行文件，用于给定测试集位置参数执行chanllenge.py生成answer.csv, 用以结果评测。
    |__ model           模型存放位置
      |__ model_01.h5   训练得到的模型，模型结构定义见f_model

    |__ f_train.py      模型训练，是不包含在提交代码中的，提供可用于自行训练得到模型测试结果。

## Dependencies

Packages

    python 3.6+
    numpy
    pandas
    scipy
    scikit-learn
    scikit-multilearn
    xgboost
    tensorflow
    keras

Hardwares

    gpu (not necessary, better if have)
    enough memory
