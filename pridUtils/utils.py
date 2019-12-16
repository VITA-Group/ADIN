import os
import time
import torch
import shutil
import GPUtil as GPU


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    return dir


def clean_file(fname):
    if os.path.isfile(fname):
        os.remove(fname)
    return fname


def getGPU():
    gpu_ids = [int(id) for id in GPU.getAvailable(order='memory', limit=8, maxMemory=0.5, maxLoad=0.8)]
    print("gpu_ids: %s" % gpu_ids)
    return gpu_ids


def scheduleGPU():
    print('\nscheduling GPU ..')
    gpu_ids = getGPU()
    while not gpu_ids:
        time.sleep(10)
        gpu_ids = getGPU()
    return gpu_ids


def scheduleGPU_id(id):
    print('\nscheduling GPU ..')
    gpu_ids = list(set(getGPU()).intersection(set(id)))
    while not gpu_ids:
        time.sleep(10)
        gpu_ids = list(set(getGPU()).intersection(set(id)))
    return gpu_ids


def setGPU(gpu_ids=None):
    if not gpu_ids:
        gpu_ids = getGPU()
    if len(gpu_ids):
        torch.cuda.set_device(gpu_ids[0])
        use_gpu = torch.cuda.is_available()
        print("set gpu: %s" % gpu_ids[0])
    else:
        raise Exception("No GPU available")

    return use_gpu


# Load model
# ---------------------------
def load_model(model, resumepath):
    print("load model from %s\n" % resumepath)
    model.load_state_dict(torch.load(resumepath))
    return model


# Save model
# ---------------------------
def save_model(model, savepath):
    print("model saved in %s\n" % savepath)
    torch.save(model.cpu().state_dict(), savepath)
    model = model.cuda()
    return model


# Save model trained with multiple GPU
# ---------------------------
def save_parallel_model(model, savepath):
    print("model saved in %s\n" % savepath)
    torch.save(model.module.cpu().state_dict(), savepath)
    model = model.cuda()
    return model
