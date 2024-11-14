import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from sklearn.metrics import average_precision_score, precision_recall_curve

import torch
import torch.distributed as dist


def create_dataset(images, labels):
    dataset = []
    
    if labels is None:
        for img in images:
            sample_dict = dict()
            sample_dict['image'] = img
            dataset.append(sample_dict)
    else:
        for img, label in zip(images, labels):
            sample_dict = dict()
            sample_dict['image'] = img
            sample_dict['pred_label'] = label
            dataset.append(sample_dict)
            
    return dataset


def save_checkpoint(model, epoch, optimizer, scheduler, filename="model.pt", best_loss=0, dir_add=None, logger=None):
    state_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()
    scheduler_dict = scheduler.state_dict()
    save_dict = {"epoch": epoch, "best_loss": best_loss, "state_dict": state_dict, \
        "optimizer": optimizer_dict, "scheduler": scheduler_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
                
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
        

class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def all_reduce_mean(x):
    if not is_dist_avail_and_initialized():
        world_size =  1
    else:
        world_size = dist.get_world_size()
        
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, args.dist_url, gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=18000))
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    
    
def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def cleanup():
    dist.destroy_process_group()


def all_gather(tensor):
    return AllGatherFunction.apply(tensor)


def set_requires_grad_false(*models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = False


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)
    

def plot_regression(x, y, title, percent="None"):
    # Create a scatter plot
    plt.figure(figsize=(20, 15))
    plt.scatter(x, y, label='data points', marker='o')

    # Determine the range to plot the diagonal
    # Use the combined range of x_data and y_data for the diagonal
    min_val = min(x)
    max_val = max(x)

    # Plot the diagonal line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')

    # Optionally, set the axis limits if you want to enforce equal aspect ratio
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Add titles and labels
    plt.title(f'Plot of {title}')
    plt.xlabel('Target')
    plt.ylabel('Prediction')

    # Show grid
    #plt.grid(True)

    # Show legend
    plt.legend()
    
    # Save plot
    plt.savefig(f'regression_plot_{percent}.png', dpi=300)
    

def plot_pr_curve(targets, preds, percent="None"):
    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(targets, preds)

    # Calculate the area under the Precision-Recall curve
    average_precision = average_precision_score(targets, preds)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve (AP={average_precision:.2f})')
    plt.savefig(f'pr_curve_plot_{percent}.png', dpi=300)