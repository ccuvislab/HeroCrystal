
import torch

def reconst_weights_to_state_dict(weights, matched_cnn):
    """ Reconstruct state_dict from weights for a matched_cnn arch(vgg).
    Args: weights - weights of fully cnn model of FedMA form.
          matched_cnn - matched model arch.
    Return : state_dict for a torch model.
    """

    new_state_dict = {}

    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):

        if "weight" in key_name:
            temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
        elif "bias" in key_name:
            temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    
    return new_state_dict


def vgg_freeze_to(vgg, layer_to_freeze):
    """ Freeze the VGG's weights before "layer_to_freeze"
    Args: vgg - vgg model.
          layer_to_freeze - layer number to freeze.
    Return: None
    """
    
    ### *2 due to a single layer is composed of weights + bias.
    for layer in range(2*layer_to_freeze):
        for p in vgg[layer].parameters(): p.requires_grad = False


def rcnn_intermediate(model, x):
    """ Get intermediate results of specifc model.
        Note: This is NOT a generalized function for all torch models,
              due to the different arch. of models.
    """
    from torch.nn import MaxPool2d, AdaptiveAvgPool2d
    
    # forward the features layers of VGG.
#     for l in list(model.RCNN_base.modules())[0]:
    for l in list(model.features.modules())[0]:
        x = l(x)
        
    x = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)   
    
    x = AdaptiveAvgPool2d(output_size=(7, 7))(x)
    
    # flatten for FC layers.
    x = x.view(x.shape[0], -1)

#     # go through FC layers.
#     for l in list(model.RCNN_top.modules())[0]:
#         x = l(x)

    return x


def get_features(frcnn_model, imgs, batch_sz=64):
    """ Get intermediate result from FRCNN model.
    Args : frcnn_model - torch model.
           imgs - img list with proper format.
    """
    
    outputs = []
    n_batch = len(imgs)//batch_sz
    res_batch = len(imgs)%batch_sz    

    with torch.no_grad():
        for i in range(n_batch):
            in_batch = torch.cat(imgs[i*batch_sz:(i+1)*batch_sz]).to('cuda')
            x = rcnn_intermediate(frcnn_model, in_batch)
            in_batch.to('cpu')
            outputs.append(x)

        # last incomplete batch.
        in_batch = torch.cat(imgs[-res_batch:]).to('cuda')
        x = rcnn_intermediate(frcnn_model, in_batch)
        in_batch.to('cpu')
        outputs.append(x)

    outputs = torch.cat(outputs)
    X = outputs.to('cpu').numpy()
    
    return X


def within_cluster_dispersion(x, n_cluster):
    """ Calculate the pooled within-cluster sum of squares around the cluster means.
        Wk = 1/(2*n)*sum(Dr), where Dr = sum(d(xi, xj)) for xi, xj in the cluster.
        Wk is equivalent to inertia in kmeans.
    Args: x - data, of shape (n, feature_dim)
          n_cluster - number of cluster
    Return: Wk
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n_x = x.shape[0] # number of data.

    kmeans = KMeans(n_cluster)
    kmeans.fit(x)
    
    return kmeans.inertia_/n_x


def gap_stats(x, n_cluster=10, n_samples=5):
    """ Calculate the pooled within-cluster sum of squares around the cluster means.
        Wk = 1/(2*n)*sum(Dr), where Dr = sum(d(xi, xj)) for xi, xj in the cluster.
    Args: x - data, of shape (n, feature_dim)
          n_cluster - number of cluster
          n_samples - to estimate the gap statistics (seems not necessary for Wk)
    Return: Wk
    """
    
    import numpy as np
    from sklearn.cluster import KMeans
    
    def calc_Dr(x):
        """ Helper for calculating Dr.
        Args: x - data of shape (n, feature_dim)
        Return: Dr (as def in the original paper)
        """
        calc_d = lambda x: np.sqrt(np.square(x).sum(axis=-1)).sum()
        Dr = np.sum([calc_d(x- xi) for xi in x])
        return Dr
    
    
    n_x = x.shape[0] # number of data.
    # kmeans for clustering.
    kmeans = KMeans(n_cluster)
    kmeans.fit(x)
    x_pred = kmeans.predict(x)
    
    Wk = []

    # accumulate the within-cluster sum of squares.
    for idx in range(n_cluster):
        x_tmp = x[x_pred == idx]
        Dr = calc_Dr(x_tmp)
        Wk.append(0.5*Dr/n_x)
    
    return np.mean(Wk)


def getWeight(test_images, model_list, args):
    """  calculate the dynamic attention weights for FedWCD.
    Args: test_images - test images as a pickle file.
          model_lsit  - models.
    """
    
    wk_list = []
    
    for fasterRCNN in model_list:
        
        if args.mGPUs: 
            fasterRCNN = fasterRCNN.module
        
        X = get_features(fasterRCNN, test_images, args.batch_size)/255.0
        
        wk_value = within_cluster_dispersion(X, n_cluster = args.k)
        wk_list.append(wk_value)
        
        print(wk_value)
    
    return wk_list 


def weights_for_avg(args, 
                    wk_list_prev,
                    round_idx,
                    parties,
                    model_list):
    
    from scipy.special import softmax
    import pickle

    if args.wkFedAvg:
        # read test image pickle file
        testimg_pickle_path = args.testimg_pickle_path #'testimg2252.pkl'

        with open(testimg_pickle_path, 'rb') as handle:
            test_images = pickle.load(handle)
        # get within class dispersion        
        
        wk_list_curr = getWeight(test_images, model_list, args)
        
        if round_idx==1:
            wk_diff = wk_list_curr
        else:
            wk_diff=[]
            for list1_c, list2_p in zip(wk_list_prev, wk_list_curr):        
                wk_diff.append(list1_c - list2_p)

        print('diff={}'.format(wk_diff))

        wk_ratio = softmax(wk_diff).tolist()    
        print('wk_ratio={}'.format(wk_ratio))

        #wk_ratio = [x / sum(wk_diff) for x in wk_diff]
        #keep wk to previous
        wk_list_prev = wk_list_curr
    else:
        wk_ratio =  [1] * parties 
        wk_ratio = [x / parties for x in wk_ratio]

    return wk_ratio, wk_list_prev 


def FedPer(model_list, ratio_list, mGPUs):
    """ Personalized FL share the featurizer and keep the classifier for each client.
    Args: mode_list  - models
          ratio_list - weights for models.
          mGPU - mGPUs enable or not, boolean.
    """
    
    model_tmp = [None] * len(model_list)
#     model_tmp = [None] * parties
    #optims_tmp=[None] * parties

    for idx, my_model in enumerate(model_list):
        if mGPUs:
            my_model = my_model.module
            
        model_tmp[idx] = my_model.RCNN_base.state_dict()


    for key in model_tmp[0]:    
        #print(key)
        model_avg = 0

        for idx, model_tmp_content in enumerate(model_tmp):     # add each model              
            model_avg += ratio_list[idx] * model_tmp_content[key]
            
        for i in range(len(model_tmp)):  #copy to each model            
            model_tmp[i][key] = model_avg
    
    #copy back to original model
    for i in range(len(model_list)):  
        if mGPUs:
#             model_list[i].module.RCNN_base.load_state_dict(model_tmp[i])
            model_list[i].module.features.load_state_dict(model_tmp[i])
        else:
            model_list[i].features.load_state_dict(model_tmp[i])
#             model_list[i].RCNN_base.load_state_dict(model_tmp[i])
            
    return model_list
