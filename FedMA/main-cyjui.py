from utils import *
import pickle
import copy
from sklearn.preprocessing import normalize

from matching.pfnm import layer_wise_group_descent, layer_group_descent_bottom_up
from matching.pfnm import block_patching, patch_weights

from matching.gaus_marginal_matching import match_local_atoms
from combine_nets import compute_pdm_matching_multilayer, compute_iterative_pdm_matching
from matching_performance import compute_model_averaging_accuracy, compute_pdm_cnn_accuracy, compute_pdm_vgg_accuracy, compute_full_cnn_accuracy

from helper_cyjui import *

from vgg import matched_vgg11

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

args_logdir = "logs/cifar10"
#args_dataset = "cifar10"
args_datadir = "./data/cifar10"
args_init_seed = 0
args_net_config = [3072, 100, 10]
#args_partition = "hetero-dir"
args_partition = "homo"
args_experiment = ["u-ensemble", "pdm"]
args_trials = 1
#args_lr = 0.01
args_epochs = 5
args_reg = 1e-5
args_alpha = 0.5
args_communication_rounds = 5
args_iter_epochs=None

args_pdm_sig = 1.0
args_pdm_sig0 = 1.0
args_pdm_gamma = 1.0


def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped


def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped


def local_train(nets, args, net_dataidx_map, device="cpu"):
    # save local dataset
    local_datasets = []
    for net_id, net in nets.items():
        
        if args.retrain:
            dataidxs = net_dataidx_map[net_id]
            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
            train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 32)

            local_datasets.append((train_dl_local, test_dl_local))
            # switch to global test set here
            ## cyjui.
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.epochs, args.lr, args, device=device)
#             trainacc, testacc = 0, 0
    
            # saving the trained models here
            save_model(net, net_id)
        else:
            load_model(net, net_id, device=device)

    nets_list = list(nets.values())
    return nets_list


def BBP_MAP(nets_list, model_meta_data, 
            layer_type, net_dataidx_map, 
            averaging_weights, args, 
            device="cpu"):
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    n_classes = args_net_config[-1]
    it=5
    ### commented 21/11/17.
#     sigma  = args_pdm_sig 
#     sigma0 = args_pdm_sig0
#     gamma  = args_pdm_gamma  
    assignments_list = []
    
    batch_weights = pdm_prepare_full_weights_cnn(models, device=device)
#     batch_weights = pdm_prepare_weights_vggs(nets_list, device=device) # 12/11/17, cyjui.
    raw_batch_weights = copy.deepcopy(batch_weights)
    
    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))
    
    
#     for idx, w in enumerate(batch_weights[0]):
#         print("{:}, {:}".format(idx, w.shape))

    ### calculate the freq. of classes.
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gamma  = 7.0
    sigma  = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None
    
    # cyjui.
#     print("============== BBP_MAP layers ===============\n")
#     print(n_layers)
    
    for layer_index in range(1, n_layers):
        
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
#         layer_hungarian_weights, assignment, L_next = layer_group_descent_bottom_up(
             batch_weights=batch_weights, 
             layer_index=layer_index,
             sigma0_layers=sigma0, 
             sigma_layers=sigma, 
             batch_frequencies=batch_freqs, 
             it=it, 
             gamma_layers=gamma, 
             model_meta_data=model_meta_data,
             model_layer_type=layer_type,
             n_layers=n_layers,
             matching_shapes=matching_shapes,
             args=args
             )
        
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2] 
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        ### determine the first fc layer.
        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and 
                               ('conv' in type_of_prev_layer or 'features' in type_of_prev_layer))
      
        if first_fc_identifier:
            first_fc_index = layer_index
            
#         print("\nFirst FC layer {:}, {:} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(first_fc_index, first_fc_identifier))
#         print("type_of_this: {:}, type_of_prev: {:}".format(type_of_this_layer, type_of_prev_layer))
        
        matching_shapes.append(L_next)
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) 
                          for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model)
                    
                elif l_type == "fc": ### how can this happen???
                    print("+"*20 + "How can this happen???")
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                               L_next, assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])
        
#         for lid, w in enumerate(tempt_weights[0]):
#             print("BBBBB {:}, {:}".format(lid, w.shape))
        
 
        retrained_nets = []
        
        for worker_index in range(num_workers):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)

            logger.info("Re-training on local worker: {}, starting from layer: {}".
                        format(worker_index, 2 * (layer_index + 1) - 2))
            
            retrained_cnn = local_retrain_nchc((train_dl_local, test_dl_local), 
                                          tempt_weights[worker_index], args, 
                                          freezing_index=(2 * (layer_index + 1) - 2), device=device)
            
           
            retrained_nets.append(retrained_cnn)
                    
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)
#         batch_weights = pdm_prepare_weights_vggs(nets_list, device=device) # 12/11/17, cyjui.

#     print("\nOut of BBP_MAP loop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)
    
    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])
    
    return matched_weights, assignments_list


def fedma_comm(batch_weights, 
               model_meta_data, 
               layer_type, net_dataidx_map, 
                averaging_weights, args, 
                train_dl_global,
                test_dl_global,
                assignments_list,
                comm_round=2,
                device="cpu"):
    '''
    version 0.0.2
    In this version we achieve layerwise matching with communication in a blockwise style
    i.e. we unfreeze a block of layers (each 3 consecutive layers)---> retrain them ---> and rematch them
    '''
    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(batch_weights)

    matching_shapes = []
    first_fc_index = None
    gamma  = 5.0
    sigma  = 1.0
    sigma0 = 1.0

    cls_freqs = traindata_cls_counts  ### frequencies of classes.
    n_classes = 10
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    it=5

    for cr in range(comm_round):
        
        logger.info("Entering communication round: {} ...".format(cr))
        retrained_nets = []
        
        for worker_index in range(args.n_nets):
            
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)

            
            # for the "squeezing" mode, we pass assignment list wrt this worker to the `local_retrain` function
            ### This gonna reconstruct the model from global size (larger) back to the original shape.
            recons_local_net = reconstruct_local_net(batch_weights[worker_index], 
                                                     args, 
#                                                      ori_assignments=model_meta_data, 
                                                     ori_assignments=assignments_list, ### Should be the shape of original model.
                                                     worker_index=worker_index)

#             recons_local_net = batch_weights[worker_index] ### why restore to the un-reconstructed model???
        
            retrained_cnn = local_retrain_nchc((train_dl_local, test_dl_local), recons_local_net, args,
                                            mode="bottom-up", freezing_index=0, ori_assignments=None, device=device)
            
            retrained_nets.append(retrained_cnn)

        ### cyjui
        print("==== Round: {:}".format(cr))
        print(retrained_nets[0])
        
        
        # BBP_MAP step
        hungarian_weights, assignments_list = BBP_MAP_nchc(retrained_nets, model_meta_data, 
                                                           layer_type, net_dataidx_map, 
                                                           averaging_weights, args, 
                                                           traindata_cls_counts,
                                                           device=device)

        logger.info("After retraining and rematching for comm. round: {}, we measure the accuracy ...".format(cr))
        
        ### cyjui [21/12/15]
#         test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)
#         print("*"*15 + "Acc = {:}".format(test_acc))
        
#         _ = compute_full_cnn_accuracy(models,
#                                       hungarian_weights,
#                                       train_dl_global,
#                                         test_dl_global,
#                                         n_classes,
#                                         device=device,
#                                         args=args)
        
        batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]
        
        del hungarian_weights
        del retrained_nets
        
        
def calc_class_avg_weights(n_class, n_nets, traindata_cls_counts):
    """ short-hand to calculate the averaging weights(by ratio) for all clients.
    Args: n_class - # of classes.
          n_nets  - # of nets.
          traindata_cls_counts - counts by classes for all clients join training.
    
    """
    
    averaging_weights = np.zeros((args.n_nets, n_classes), dtype=np.float32)
    
    ### calculate the weights for averaging later.
    for i in range(n_classes):
        total_num_counts = 0
        worker_class_counts = [0] * n_nets
        
        for j in range(n_nets):
            if i in traindata_cls_counts[j].keys():
                total_num_counts += traindata_cls_counts[j][i]
                worker_class_counts[j] = traindata_cls_counts[j][i]
            else:
                total_num_counts += 0
                worker_class_counts[j] = 0
                
        averaging_weights[:, i] = worker_class_counts / total_num_counts
        
    return averaging_weights


def FedAVG_by_freq(batch_weights, fed_avg_freqs):
    """ FedAVG according to # of data points.
    Args: batch_weights - weights of models.(a list of weights)
          fed_avg_freqs - data point frequencies by clients.
    """
    
    ### Naive average for FedAVG.
    averaged_weights = []
    num_layers = len(batch_weights[0])
    
    for i in range(num_layers):
        avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
        averaged_weights.append(avegerated_weight)
        
    return averaged_weights


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logger.info(device)
    args = add_fit_args(argparse.ArgumentParser(description='Probabilistic Federated CNN Matching'))

    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Partitioning data")

    if args.partition != "hetero-fbs":
        y_train, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args_datadir, args_logdir, 
                                                                args.partition, args.n_nets, args_alpha, args=args)
    else:
        y_train, net_dataidx_map, traindata_cls_counts, baseline_indices = partition_data(args.dataset, args_datadir, args_logdir, 
                                                    args.partition, args.n_nets, args_alpha, args=args)

    n_classes = len(np.unique(y_train))
    
    averaging_weights = calc_class_avg_weights(n_classes, args.n_nets, traindata_cls_counts)
#     logger.info("averaging_weights: {}".format(averaging_weights))
    logger.info("Initializing nets")
    
    ### init model for clients, 
    ### model_meta_data = the shapes of the init model arch.
    ### layer_type      = a list records the model arch. (cnn or fc)
    nets, model_meta_data, layer_type = init_models(args_net_config, args.n_nets, args)
    
#     tmp = nets[0]
    
#     for param_idx, (key_name, param) in enumerate(tmp.state_dict().items()):
#         print(param_idx, key_name)
    
    
    print("====================== Before local train ===============================")
    
    ### local training stage
    nets_list = local_train(nets, args, net_dataidx_map, device=device)

    train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 32)

    # ensemble part of experiments
    logger.info("Computing Uniform ensemble accuracy")
    uens_train_acc,  uens_test_acc = 0, 0
#     uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl_global, n_classes,  uniform_weights=True, device=device)
#     uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl_global, n_classes, uniform_weights=True, device=device)

    logger.info("Uniform ensemble (Train acc): {}".format(uens_train_acc))
    logger.info("Uniform ensemble (Test acc): {}".format(uens_test_acc))
    
    
    print("====================== Before PFNM ===============================")
    # this is for PFNM
    hungarian_weights, assignments_list = BBP_MAP_nchc(nets_list, model_meta_data, 
                                                       layer_type, net_dataidx_map, 
                                                       averaging_weights, args, 
                                                       traindata_cls_counts,
                                                       device=device)

    ## averaging models 
    ## we need to switch to real FedAvg implementation 
    ## FedAvg is originally proposed at: here: https://arxiv.org/abs/1602.05629
    batch_weights = pdm_prepare_full_weights_cnn(nets_list, device=device)  # 21/11/17, cyjui.
#     batch_weights = pdm_prepare_weights_vggs(nets_list, device=device)
    
    #dataidxs = net_dataidx_map[args.rank]
    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
    logger.info("Total data points: {}".format(total_data_points))
    logger.info("Freq of FedAvg: {}".format(fed_avg_freqs))

    ### Naive average for FedAVG.
    averaged_weights = FedAVG_by_freq(batch_weights, fed_avg_freqs)
        
    print("==== Before [compute_full_cnn_accuracy] ===========================")
    print("# of Models : {:}".format(len(nets_list)))
    
    models = nets_list
    models = [model.cpu() for model in models]
    
#     _ = compute_full_cnn_accuracy(models,
#                                hungarian_weights,
#                                train_dl_global,
#                                test_dl_global,
#                                n_classes,
#                                device=device,
#                                args=args)

#     _ = compute_model_averaging_accuracy(models, 
#                                 averaged_weights, 
#                                 train_dl_global, 
#                                 test_dl_global, 
#                                 n_classes,
#                                 args)


    if args.comm_type == "fedavg":
        ########################################################
        # baseline: FedAvg: https://arxiv.org/pdf/1602.05629.pdf
        ########################################################
        # we turn to enable communication here:
        comm_init_batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]

        fedavg_comm(comm_init_batch_weights, model_meta_data, layer_type, 
                            net_dataidx_map, 
                            averaging_weights, args,
                            train_dl_global,
                            test_dl_global,
                            comm_round=args.comm_round,
                            device=device)
        
    elif args.comm_type == "fedprox":
        ##########################################################
        # baseline: FedProx: https://arxiv.org/pdf/1812.06127.pdf
        ##########################################################

        # we turn to enable communication here:
        comm_init_batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]

#         fedprox_comm(comm_init_batch_weights, model_meta_data, layer_type, 
#                             net_dataidx_map, 
#                             averaging_weights, args,
#                             train_dl_global,
#                             test_dl_global,
#                             comm_round=args.comm_round,
#                             device=device)
    elif args.comm_type == "fedma":
        
        print("="*15 + "Into FedMA" + "="*15)
        
        ### First matched (global) weights, duplicated n times.
        comm_init_batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]

        fedma_comm(comm_init_batch_weights,
                   model_meta_data, 
                   layer_type, 
                   net_dataidx_map,
                   averaging_weights, args,
                   train_dl_global,
                   test_dl_global,
                   assignments_list,
                   comm_round=args.comm_round,
                   device=device)
        
        
        print("="*15 + "End of FedMA" + "="*15)