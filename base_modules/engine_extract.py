import torch
import math
import time
import sys

import torch.distributed as dist

import pickle

from src.utils.misc import all_gather

def extract_embedding(
    config,
    model,
    data_loader,
    pooling='cls',
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
):
    model_name = config.MODEL.NAME
    model.eval()
    
    roi = config.MODEL.ROI
    num_patches = config.VIT.NUM_PATCHES
    hidden_dim = config.VIT.HIDDEN_SIZE
    
    embedding_dict = dict()
    #shape_lst = []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            data = batch_data['image'].to(device)
            batch_size = data.shape[0]
            
            #shape_lst.append(data.shape)
            
            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                if model_name == 'vit':
                    # aggregate batch and patch dimension
                    batch_data_agg = data.reshape(-1, 1, roi[0], roi[1], roi[2])
                    out, _ = model(batch_data_agg)
                    # restore orignal batch/patch dimension and average across patch dimension
                    #out = out.reshape(batch_size, num_patches, -1, hidden_dim).mean(dim=1)
                    out = out.reshape(batch_size, num_patches, -1, hidden_dim)
                else:
                    raise NotImplementedError("Unknown model: {}".format(model_name))
            
            if dist.get_rank() == 0:
                filenames = batch_data['image_meta_dict']['filename_or_obj']
                out_gather = all_gather(out).detach().cpu().numpy()
                
                for i, filename in enumerate(filenames):
                    if filename not in embedding_dict.keys():
                        if pooling == 'cls':
                            # store [cls] token embedding
                            embedding_dict[filename] = out_gather[i][:, :1, :].squeeze()
                        elif pooling == 'mean':
                            # store mean of all tokens except [cls]
                            embedding_dict[filename] = out_gather[i][:, 1:, :].mean(axis=1, keepdims=False).squeeze()
                        else:
                            raise NotImplementedError(f"Pooling {pooling} not implemented.")
                        #print(embedding_dict[filename].shape)
            
            torch.cuda.synchronize()
            
            logger.info(f"[{idx+1}/{len(data_loader)}]")
    
    print(f"DICT LEN: {len(embedding_dict)}")
    # Save embedding dictionary to a pickle file
    if dist.get_rank() == 0:
        with open(f'./embedding/10%_train_embedding_{pooling}.pkl', 'wb') as file:
            pickle.dump(embedding_dict, file)
        
    # with open(f'./embedding_v2/shape_lst.pkl', 'wb') as file:
    #     pickle.dump(shape_lst, file)
    