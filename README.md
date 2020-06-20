# Instruction to re-implement our results 

## Requirements
    - pytorch(我们的是1.2.0)
    - python3

## first task: 1 day
使用我们的预训练模型直接测试

`python reimplement.py --exp_name first_task_pretrained --test`

从头训练模型，并测试

`python reimplement.py --exp_name first_task`

将每个模型的结果文件合并，得到当前任务的最终结果文件

`python toy_experiments.py --name first_task(or first_task_pretrained) --out_name first_task(or first_task_pretrained)`

## second task: 20 day
使用我们的预训练模型直接测试

`python reimplement.py --exp_name second_task_pretrained --test`

从头训练模型，并测试

`python reimplement.py --exp_name second_task`

将每个模型的结果文件合并，得到当前任务的最终结果文件

`python toy_experiments.py --name second_task(or second_task_pretrained) --out_name second_task(or second_task_pretrained)`

## third task: 60 day
使用我们的预训练模型直接测试

`python reimplement.py --exp_name third_task_pretrained --test`

从头训练模型，并测试

`python reimplement.py --exp_name third_task`

将每个模型的结果文件合并，得到当前任务的最终结果文件

`python toy_experiments.py --name third_task(or third_task_pretrained) --out_name third_task(or third_task_pretrained)`

## 不同超参数，结果文件合并
我们的代码，需要您手动，将不同参数的结果合并，得到最终的结果。

- 需要在results文件夹，新建一个final文件夹
- 将results/first_task文件夹里面1day后缀的csv文件，复制到final文件夹，
- second_task, third_task也分别复制20day和60day后缀的csv文件, 
- 然后执行python toy_experiments.py --name final --out_name final
- 主目录下生成final.csv，提交到系统。

在整理代码，重新训练测试复现结果时，发现与最初的代码，有一定出入，希望您可以将每个超参数下(即所有任务使用同一套参数)，模型的测试结果都测试一下，如果使用上述提到的命令，则应该为`first_task.csv, second_task.csv, third_task.csv`。