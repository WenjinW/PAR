# Task Difficulty Aware Parameter Allocation & Regularization for Lifelong Learning
This is the source code of Paper: Task Difficulty Aware Parameter Allocation & Regularization for Lifelong Learning. ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Task_Difficulty_Aware_Parameter_Allocation__Regularization_for_Lifelong_Learning_CVPR_2023_paper.html), [arXiv](https://arxiv.org/abs/2304.05288), [YouTube Video](https://www.youtube.com/watch?v=R0jA9rHxIWI))

## Enviroment
We provide the conda enviroment file [here](./CondaEnv.yml)

## Run experiment
```bash
source run_files/[run_file_name.sh]
```

### Example: run PAR on CIFAR100-10
```bash
sh run_files/cifar100-10/par2.sh
```
The result is as follows：
![image](results.png)

## Code
in /src

## Citation
Please cite the paper if you use the code in this repo.
```latex
@inproceedings{wang2023task,
  title={Task Difficulty Aware Parameter Allocation \& Regularization for Lifelong Learning},
  author={Wang, Wenjin and Hu, Yunqing and Chen, Qianglong and Zhang, Yin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7776--7785},
  year={2023}
}
```
