import json
import numpy as np

def compute_backward_transfer(acc):
    task_num = acc.shape[0]
    backward_acc = 0.0
    for idx in range(task_num):
        backward_acc += (acc[task_num-1][idx]- acc[idx][idx])
    mean_backward_acc = backward_acc / task_num
    print(task_num)
    print("mean forget acc", mean_backward_acc)
    return mean_backward_acc.item()

def compute_forward_transfer(acc):
    task_num = acc.shape[0]
    forward_acc = 0.0
    for idx in range(1, task_num):
        forward_acc += (acc[idx-1][idx] - 1.0/task_num)
    print(forward_acc)
    mean_forward_acc = forward_acc / (task_num-1)
    print("mean forward acc", mean_forward_acc)
    return mean_forward_acc.item()

def compute_newtask_performance(acc):
    task_num = acc.shape[0]
    new_acc = 0.0
    for idx in range(task_num):
        new_acc += acc[idx][idx]
    mean_new_acc = new_acc / task_num
    
    return mean_new_acc.item()


def get_acc_metric(acc_m, prefix='test'):
    acc_m = np.array(acc_m)

    return {
        f"{prefix}_ap": np.mean(acc_m[-1, :]).item(),
        f"{prefix}_bwt": compute_backward_transfer(acc_m),
        f"{prefix}_fwt": compute_forward_transfer(acc_m),
        f"{prefix}_new": compute_newtask_performance(acc_m),
        f"{prefix}_acc_m": acc_m.tolist(),
    }


def get_metric(filename, method):
    
    if type(filename) is str:
        with open(filename) as f:
            result = json.load(f)
    else:
        result = filename
    
    metrics = {
        'total_model_size(M)': result['model_size'][-1].item() if method != 'pn' else result['model_size'][-1],
        'model_size(M)': result['model_size'],
        'time(h)': result['time'],
    }
    metrics.update(get_acc_metric(result['val_acc_m'], "val"))
    metrics.update(get_acc_metric(result['test_acc_m'], "test"))


    if 'info' in result.keys():
        metrics['info'] = result['info']
    
    # for k, v in metrics.items():
    #     if isinstance(v, np.ndarray):
    #         metrics[k] = v.tolist()
    #         print(k, type(v))

    return metrics

if __name__ == "__main__":
    filename = './res/core50_AutoCLL_0_001_<built-in function id>.json'
    result = get_metric(filename)
    print(result)