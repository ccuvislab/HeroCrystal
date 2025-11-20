
### [21/12/29, cyjui]
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils import *
import pickle
import copy
from sklearn.preprocessing import normalize

from matching.pfnm import layer_wise_group_descent, layer_group_descent_bottom_up
from matching.pfnm import block_patching, patch_weights

### [cyjui, 21/12/29]
from matching.pfnm import layer_wise_group_descent_full_cnn
from matching.pfnm import block_patching_frcnn

from matching.gaus_marginal_matching import match_local_atoms
from combine_nets import compute_pdm_matching_multilayer, compute_iterative_pdm_matching
from matching_performance import compute_model_averaging_accuracy, compute_pdm_cnn_accuracy, compute_pdm_vgg_accuracy, compute_full_cnn_accuracy

from vgg import matched_vgg11, VGGConvBlocks

from vgg import vgg16_rcnn, matched_vgg16_no_FC ### cyjui, 21/12/30.


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.WARN)

def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped


def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped

def pdm_prepare_full_weights_nchc(nets, device="cpu"):
    """
    we extract all weights of the conv nets out here:
    """
    weights = []
    for net_i, net in enumerate(nets):
        
        net_weights = []
        statedict = net.state_dict()

        for param_id, (k, v) in enumerate(statedict.items()):
            
            if device != "cpu":
                v = v.cpu()
                
            if 'fc' in k or 'classifier' in k:
                if 'weight' in k:
                    net_weights.append(v.numpy().T)
                else:
                    net_weights.append(v.numpy())
            elif 'conv' in k or 'features' in k:
                if 'weight' in k:
                    _weight_shape = v.size()
                    if len(_weight_shape) == 4:
                        net_weights.append(v.numpy().reshape(_weight_shape[0],
                                                             _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                    else:
                        pass
                else:
                    net_weights.append(v.numpy())        
                        
        weights.append(net_weights)  
        
    return weights


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lenet', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--retrain_lr', type=float, default=0.1, metavar='RLR',
                        help='learning rate using in specific for local network retrain (default: 0.01)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1, metavar='FLR',
                        help='learning rate using in specific for fine tuning the softmax layer on the data center (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained in a training process')
    parser.add_argument('--retrain_epochs', type=int, default=10, metavar='REP',
                        help='how many epochs will be trained in during the locally retraining process')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, metavar='FEP',
                        help='how many epochs will be trained in during the fine tuning process')
    parser.add_argument('--partition_step_size', type=int, default=6, metavar='PSS',
                        help='how many groups of partitions we will have')
    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')
    parser.add_argument('--partition_step', type=int, default=0, metavar='PS',
                        help='how many sub groups we are going to use for a particular training process')                          
    parser.add_argument('--n_nets', type=int, default=2, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--oneshot_matching', type=bool, default=False, metavar='OM',
                        help='if the code is going to conduct one shot matching')
    parser.add_argument('--retrain', type=bool, default=False, 
                            help='whether to retrain the model or load model locally')
    parser.add_argument('--rematching', type=bool, default=False, 
                            help='whether to recalculating the matching process (this is for speeding up the debugging process)')
    parser.add_argument('--comm_type', type=str, default='layerwise', 
                            help='which type of communication strategy is going to be used: layerwise/blockwise')    
    parser.add_argument('--comm_round', type=int, default=10, 
                            help='how many round of communications we shoud use')  
    args = parser.parse_args()
    return args


def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".
                    format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "moderate-cnn":
        matched_cnn = ModerateCNN()

    new_state_dict = {}
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
        
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]

#     optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), 
#                                      lr=0.001, weight_decay=0.0001, amsgrad=True)
    optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), 
                                    lr=args.retrain_lr, momentum=0.9, weight_decay=0.0001)
    
    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: %d' % len(train_dl_local))
    logger.info('n_test: %d' % len(test_dl_local))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)


    for epoch in range(args.retrain_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)
            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()

        #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn

# def fedma_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map, 
#                             averaging_weights, args, 
#                             train_dl_global,
#                             test_dl_global,
#                             assignments_list,
#                             comm_round=2,
#                             device="cpu"):
#     '''
#     version 0.0.2
#     In this version we achieve layerwise matching with communication in a blockwise style
#     i.e. we unfreeze a block of layers (each 3 consecutive layers)---> retrain them ---> and rematch them
#     '''
#     n_layers = int(len(batch_weights[0]) / 2)
#     num_workers = len(batch_weights)

#     matching_shapes = []
#     first_fc_index = None
#     gamma = 5.0
#     sigma = 1.0
#     sigma0 = 1.0

#     cls_freqs = traindata_cls_counts
#     n_classes = 10
#     batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
#     it=5

#     for cr in range(comm_round):
#         logger.info("Entering communication round: {} ...".format(cr))
#         retrained_nets = []
        
#         for worker_index in range(args.n_nets):
            
#             dataidxs = net_dataidx_map[worker_index]
#             train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)

#             # for the "squeezing" mode, we pass assignment list wrt this worker to the `local_retrain` function
#             recons_local_net = reconstruct_local_net(batch_weights[worker_index], 
#                                                      args, 
#                                                      ori_assignments=assignments_list, worker_index=worker_index)
            
#             retrained_cnn = local_retrain((train_dl_local, test_dl_local), recons_local_net, args,
#                                             mode="bottom-up", freezing_index=0, ori_assignments=None, device=device)
#             retrained_nets.append(retrained_cnn)

#         ### cyjui
#         print("==== Round: {:}, layer: {:} ==== ".format(cr, i))
        
#         # BBP_MAP step
#         hungarian_weights, assignments_list = BBP_MAP(retrained_nets, model_meta_data, 
#                                                       layer_type, net_dataidx_map, 
#                                                       averaging_weights, args, device=device)

#         logger.info("After retraining and rematching for comm. round: {}, we measure the accuracy ...".format(cr))
#         _ = compute_full_cnn_accuracy(models,
#                                    hungarian_weights,
#                                    train_dl_global,
#                                    test_dl_global,
#                                    n_classes,
#                                    device=device,
#                                    args=args)
        
#         batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]
#         del hungarian_weights
#         del retrained_nets


def fedavg_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
                            averaging_weights, args,
                            train_dl_global,
                            test_dl_global,
                            comm_round=2,
                            device="cpu"):

    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    for cr in range(comm_round):
        retrained_nets = []
        logger.info("Communication round : {}".format(cr))
        for worker_index in range(args.n_nets):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
            
            # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
            retrained_cnn = local_retrain_fedavg((train_dl_local,test_dl_local), batch_weights[worker_index], args, device=device)
            
            retrained_nets.append(retrained_cnn)
            
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
        averaged_weights = []
        num_layers = len(batch_weights[0])
        
        for i in range(num_layers):
            avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
            averaged_weights.append(avegerated_weight)

        _ = compute_full_cnn_accuracy(None,
                            averaged_weights,
                            train_dl_global,
                            test_dl_global,
                            n_classes,
                            device=device,
                            args=args)
        batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
        del averaged_weights


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args, device="cpu"):
    
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.dataset == "cinic10":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    losses, running_losses = [], []

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if args.dataset == "cinic10":
            scheduler.step()

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def local_retrain_dummy(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None, device="cpu"):
    """
    FOR FPNM
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        if mode not in ("block-wise", "squeezing"):
            num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], 
                           weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
            input_dim = weights[12].shape[0]
            hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

            input_dim = weights[12].shape[0]
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            # so at here we need to generate a unfreezing list:
            __unfreezing_list = []
            for fi in freezing_index:
                __unfreezing_list.append(2*fi-2)
                __unfreezing_list.append(2*fi-1)

            # we need to do two changes here:
            # i) switch the number of filters in the freezing indices block to the original size
            # ii) cut the correspoidng color channels
            __fixed_indices = set([i*2 for i in range(6)]) # 0, 2, 4, 6, 8, 10
            dummy_model = ModerateCNN()

            num_filters = []
            for pi, param in enumerate(dummy_model.parameters()):
                if pi in __fixed_indices:
                    if pi in __unfreezing_list:
                        num_filters.append(param.size()[0])
                    else:
                        num_filters.append(weights[pi].shape[0])
            del dummy_model
            logger.info("################ Num filters for now are : {}".format(num_filters))
            # note that we hard coded index of the last conv layer here to make sure the dimension is compatible
            if freezing_index[0] != 6:
            #if freezing_index[0] not in (6, 7):
                input_dim = weights[12].shape[0]
            else:
                # we need to estimate the output shape here:
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                #estimated_shape = (estimated_output[1], estimated_output[2], estimated_output[3])
                input_dim = estimated_output.view(-1).size()[0]

            if (freezing_index[0] <= 6) or (freezing_index[0] > 8):
                hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
            else:
                dummy_model = ModerateCNN()
                for pi, param in enumerate(dummy_model.parameters()):
                    if pi == 2*freezing_index[0] - 2:
                        _desired_shape = param.size()[0]
                if freezing_index[0] == 7:
                    hidden_dims = [_desired_shape, weights[14].shape[1]]
                elif freezing_index[0] == 8:
                    hidden_dims = [weights[12].shape[1], _desired_shape]
        elif mode == "squeezing":
            pass


        if args.dataset in ("cifar10", "cinic10"):
            if mode == "squeezing":
                matched_cnn = ModerateCNN()
            else:
                matched_cnn = ModerateCNNContainer(3,
                                                    num_filters, 
                                                    kernel_size=3, 
                                                    input_dim=input_dim, 
                                                    hidden_dims=hidden_dims, 
                                                    output_dim=10)
        elif args.dataset == "mnist":
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=10)
    
    new_state_dict = {}
    model_counter = 0
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    #if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"):
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(__unfreezing_list[-1]+1) # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]

    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, 
                              weight_type="conv_weight", slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the 
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to 
               slice the color channel 
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
        return res_weight

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            # we need to reconstruct the weights here s.t.
            # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
            # ii) each neuron comes from the corresponding global neuron
            _matched_weight = weights[param_idx]
            _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
            #
            # we now use this `_slice_dim` for both conv layers and fc layers
            if __non_loading_indices.index(param_idx) != 2:
                _slice_dim = "filter" # please note that for biases, it doesn't really matter if we're going to use filter or channel
            else:
                _slice_dim = "channel"

            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="conv_weight", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)                   
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    if freezing_index[0] != 6:
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight)}
                    else:
                        # that's for handling the first fc layer that is connected to the conv blocks
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=estimated_output.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="first_fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight.T)}            
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
        else:
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    return matched_cnn


def local_retrain_fedprox(local_datasets, weights, mu, args, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    Implementing FedProx Algorithm from: https://arxiv.org/pdf/1812.06127.pdf
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "moderate-cnn":
        matched_cnn = ModerateCNN()

    new_state_dict = {}
    # handle the conv layers part which is not changing
    global_weight_collector = []
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                global_weight_collector.append(torch.from_numpy(weights[param_idx].reshape(param.size())).to(device))
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                global_weight_collector.append(torch.from_numpy(weights[param_idx]).to(device))
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                global_weight_collector.append(torch.from_numpy(weights[param_idx].T).to(device))
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                global_weight_collector.append(torch.from_numpy(weights[param_idx]).to(device))
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]

    #optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001, weight_decay=0.0001, amsgrad=True)
    optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9, weight_decay=0.0001)
    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: {}'.format(len(train_dl_local)))
    logger.info('n_test: {}'.format(len(test_dl_local)))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    for epoch in range(args.retrain_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)
            
            #########################we implement FedProx Here###########################
            fed_prox_reg = 0.0
            for param_index, param in enumerate(matched_cnn.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg
            ##############################################################################

            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()

        #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn


def fedprox_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
                            averaging_weights, args,
                            train_dl_global,
                            test_dl_global,
                            comm_round=2,
                            device="cpu"):

    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    for cr in range(comm_round):
        retrained_nets = []
        logger.info("Communication round : {}".format(cr))
        for worker_index in range(args.n_nets):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
            
            # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
            # local_retrain_fedprox(local_datasets, weights, mu, args, device="cpu")
            retrained_cnn = local_retrain_fedprox((train_dl_local,test_dl_local), batch_weights[worker_index], mu=0.001, args=args, device=device)
            
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
        averaged_weights = []
        num_layers = len(batch_weights[0])
        
        for i in range(num_layers):
            avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
            averaged_weights.append(avegerated_weight)

        _ = compute_full_cnn_accuracy(None,
                            averaged_weights,
                            train_dl_global,
                            test_dl_global,
                            n_classes,
                            device=device,
                            args=args)
        batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
        del averaged_weights


# def oneshot_matching(nets_list, model_meta_data, layer_type, net_dataidx_map, 
#                             averaging_weights, args, 
#                             device="cpu"):
#     # starting the neural matching
#     models = nets_list
#     cls_freqs = traindata_cls_counts
#     n_classes = args_net_config[-1]
#     it=5
#     sigma=args_pdm_sig 
#     sigma0=args_pdm_sig0
#     gamma=args_pdm_gamma
#     assignments_list = []
    
#     batch_weights = pdm_prepare_full_weights_cnn(models, device=device)
#     raw_batch_weights = copy.deepcopy(batch_weights)
    
#     logging.info("=="*15)
#     logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

#     batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
#     res = {}
#     best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

#     gamma = 7.0
#     sigma = 1.0
#     sigma0 = 1.0

#     n_layers = int(len(batch_weights[0]) / 2)
#     num_workers = len(nets_list)
#     matching_shapes = []

#     first_fc_index = None

#     for layer_index in range(1, n_layers):
#         layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
#              batch_weights=batch_weights, 
#              layer_index=layer_index,
#              sigma0_layers=sigma0, 
#              sigma_layers=sigma, 
#              batch_frequencies=batch_freqs, 
#              it=it, 
#              gamma_layers=gamma, 
#              model_meta_data=model_meta_data,
#              model_layer_type=layer_type,
#              n_layers=n_layers,
#              matching_shapes=matching_shapes,
#              args=args
#              )
#         assignments_list.append(assignment)
        
#         # iii) load weights to the model and train the whole thing
#         type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
#         if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
#             l_type = "conv"
#         elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
#             l_type = "fc"

#         type_of_this_layer = layer_type[2 * layer_index - 2]
#         type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
#         first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and ('conv' in type_of_prev_layer or 'features' in type_of_this_layer))
        
#         if first_fc_identifier:
#             first_fc_index = layer_index
        
#         matching_shapes.append(L_next)
#         #tempt_weights = [batch_weights[0][i] for i in range(2 * layer_index - 2)] + [copy.deepcopy(layer_hungarian_weights) for _ in range(num_workers)]
#         tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w in range(num_workers)]

#         # i) permutate the next layer wrt matching result
#         for worker_index in range(num_workers):
#             if first_fc_index is None:
#                 if l_type == "conv":
#                     patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
#                                         L_next, assignment[worker_index], 
#                                         layer_index+1, model_meta_data,
#                                         matching_shapes=matching_shapes, layer_type=l_type,
#                                         dataset=args.dataset, network_name=args.model)
#                 elif l_type == "fc":
#                     patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
#                                         L_next, assignment[worker_index], 
#                                         layer_index+1, model_meta_data,
#                                         matching_shapes=matching_shapes, layer_type=l_type,
#                                         dataset=args.dataset, network_name=args.model).T

#             elif layer_index >= first_fc_index:
#                 patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index]).T

#             tempt_weights[worker_index].append(patched_weight)

#         # ii) prepare the whole network weights
#         for worker_index in range(num_workers):
#             for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
#                 tempt_weights[worker_index].append(batch_weights[worker_index][lid])

#         retrained_nets = []
#         for worker_index in range(num_workers):
#             dataidxs = net_dataidx_map[worker_index]
#             train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)
#             logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index, 2 * (layer_index + 1) - 2))
#             retrained_cnn = local_retrain_dummy((train_dl_local,test_dl_local), tempt_weights[worker_index], args, 
#                                             freezing_index=(2 * (layer_index + 1) - 2), device=device)
#             retrained_nets.append(retrained_cnn)
#         batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

#     ## we handle the last layer carefully here ...
#     ## averaging the last layer
#     matched_weights = []
#     num_layers = len(batch_weights[0])

#     with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
#         pickle.dump(batch_weights, weights_file)

#     last_layer_weights_collector = []

#     for i in range(num_workers):
#         # firstly we combine last layer's weight and bias
#         bias_shape = batch_weights[i][-1].shape
#         last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
#         last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
#         # the directed normalization doesn't work well, let's try weighted averaging
#         last_layer_weights_collector.append(last_layer_weights)

#     last_layer_weights_collector = np.array(last_layer_weights_collector)
    
#     avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

#     for i in range(n_classes):
#         avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
#         for j in range(num_workers):
#             avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
#         avg_last_layer_weight[:, i] = avg_weight_collector

#     #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
#     for i in range(num_layers):
#         if i < (num_layers - 2):
#             matched_weights.append(batch_weights[0][i])

#     matched_weights.append(avg_last_layer_weight[0:-1, :])
#     matched_weights.append(avg_last_layer_weight[-1, :])
#     return matched_weights, assignments_list


def reconstruct_local_net(weights, args, ori_assignments=None, worker_index=0):
    
    def __reconstruct_weights(weight, assignment, 
                              layer_ori_shape, 
                              matched_num_filters=None, 
                              weight_type="conv_weight", slice_dim="filter"):
        
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight  = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight    = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
                
        elif weight_type == "bias":
            res_weight = weight[assignment]
            
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :]
                
        return res_weight
    
    ### select base model type.
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".
                    format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        
        ### This will use the global model arch. (larger than base model)
#         matched_shapes = [w.shape for w in weights]
#         matched_cnn = matched_vgg11(matched_shapes=matched_shapes)

        ### Original net arch.
        matched_cnn = vgg11()
        
        shape_estimator = matched_cnn.features

        dummy_input = torch.rand(1, 3, 224//2, 224//2)

        estimated_output = shape_estimator(dummy_input)
        
        print("Estimated output shape: {:}".format(estimated_output))
        
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        matched_cnn = ModerateCNN()

        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], 
                       weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        
        # we need to estimate the output shape here:
        shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
        dummy_input = torch.rand(1, 3, 32, 32)
        estimated_output = shape_estimator(dummy_input)
        input_dim = estimated_output.view(-1).size()[0]    
    
    print("##### In reconsturct_local_net #####")
    for idx, w in enumerate(weights):
        print(idx, w.shape)
    
    print("##### matched model #####")
    print(matched_cnn)

    reconstructed_weights = []
    
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        
        _matched_weight = weights[param_idx]  ### This is from global model, it might larger than the local arch.
        
        print("layer: {:}, matched_w: {:}, cnn_w: {:}".format(param_idx, param.size(), _matched_weight.shape))
        
        ### dealing with input layer.
        if param_idx < 1: # we need to handle the 1st conv layer specificly since the color channels are aligned
            _assignment = ori_assignments[int(param_idx / 2)][worker_index]
            _res_weight = __reconstruct_weights(weight=_matched_weight, 
                                                assignment=_assignment, 
                                                layer_ori_shape=param.size(), 
                                                matched_num_filters=None,
                                                weight_type="conv_weight", slice_dim="filter")
            
            reconstructed_weights.append(_res_weight)

        elif (param_idx >= 1) and (param_idx < len(weights) -2):
            
            if "bias" in key_name: # the last bias layer is already aligned so we won't need to process it
                _assignment = ori_assignments[int(param_idx / 2)][worker_index]
                _res_bias = __reconstruct_weights(weight=_matched_weight, 
                                                  assignment=_assignment, 
                                                  layer_ori_shape=param.size(), 
                                                  matched_num_filters=None,
                                                  weight_type="bias", 
                                                  slice_dim=None)
                
                reconstructed_weights.append(_res_bias)

            elif "conv" in key_name or "features" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment  = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                
                _matched_num_filters = weights[param_idx - 2].shape[0]
                _layer_ori_shape = list(param.size())
                _layer_ori_shape[0] = _matched_weight.shape[0]
                
                print("param_idx: {:}, _matched_num_filters: {:}".format(param_idx, _matched_num_filters))

                _temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                    layer_ori_shape=_layer_ori_shape, matched_num_filters=_matched_num_filters,
                                                    weight_type="conv_weight", slice_dim="channel")

                _res_weight = __reconstruct_weights(weight=_temp_res_weight, assignment=cur_assignment, 
                                                    layer_ori_shape=param.size(), matched_num_filters=None,
                                                    weight_type="conv_weight", slice_dim="filter")
                
                reconstructed_weights.append(_res_weight)

            elif "fc" in key_name or "classifier" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                _matched_num_filters = weights[param_idx - 2].shape[0]

                if param_idx != 16: ### cyjui for vgg [21/12/15]
#                 if param_idx != 12: # this is the index of the first fc layer
                    #logger.info("%%%%%%%%%%%%%%% prev assignment length: {}, 
                    #cur assignmnet length: {}".format(len(prev_assignment), len(cur_assignment)))
                    temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="fc_weight", slice_dim="channel")

                    _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                                        weight_type="fc_weight", slice_dim="filter")

                    reconstructed_weights.append(_res_weight.T)
                else:
                    # that's for handling the first fc layer that is connected to the conv blocks
                    temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment,
                                                            layer_ori_shape=estimated_output.size(), 
                                                            matched_num_filters=_matched_num_filters,
                                                            weight_type="first_fc_weight", slice_dim=None)

                    _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                                        weight_type="fc_weight", slice_dim="filter")
                    reconstructed_weights.append(_res_weight.T)
                    
        elif param_idx  == len(weights) - 2:
            # this is to handle the weight of the last layer
            prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
            _res_weight = _matched_weight[prev_assignment, :]
            reconstructed_weights.append(_res_weight)
        elif param_idx  == len(weights) - 1:
            reconstructed_weights.append(_matched_weight)

    return reconstructed_weights


# BAD practice.
args_datadir = "./data/cifar10"

def BBP_MAP_nchc(nets_list, model_meta_data, 
                 layer_type, net_dataidx_map, 
                 averaging_weights, args,
                 traindata_cls_counts,
                 device="cpu"): 
    
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    n_classes = 10   ### args_net_config[-1]  ### BAD practice.
    it=5
    
    assignments_list = []
    
    ### prepare the weights of the existing models. (in fact, pdm_prepare_full_weights is in general form)
    batch_weights = pdm_prepare_full_weights_nchc(models, device=device)
    
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))
    
    ### calculate the freq. of classes.
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
#     res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    ### parameters for weights matching algorithm.
    gamma  = 7.0
    sigma  = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)  ### bad assumption?? assume all layers of the form [weights w/ bias].
    
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None
    
    ### matching layer by layer.
    for layer_index in range(1, n_layers):
        
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
#         layer_hungarian_weights, assignment, L_next = layer_group_descent_bottom_up(
             batch_weights = batch_weights, 
             layer_index   = layer_index,
             sigma0_layers = sigma0, 
             sigma_layers  = sigma, 
             batch_frequencies = batch_freqs, 
             it=it, 
             gamma_layers     = gamma, 
             model_meta_data  = model_meta_data,
             model_layer_type = layer_type,
             n_layers = n_layers,
             matching_shapes = matching_shapes,
             args = args
             )
        
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2] 
        
        
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        ###  Determine the 1st fc layer ###########################
        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and 
                               ('conv' in type_of_prev_layer or 'features' in type_of_prev_layer))
        if first_fc_identifier:
            first_fc_index = layer_index
        ############################################################
        
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
                elif l_type == "fc":
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
                    
        batch_weights = pdm_prepare_full_weights_nchc(retrained_nets, device=device)
        
    ### ==========  end of layer wise matching. ==============

#     print("\nOut of BBP_MAP loop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    ### save weights of all cleints.
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

    ### weighted average according to the population of classes.
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
    
    """
        matched_weights : identical to batch_weights(matched retrained model) except the last weights & bias.
        assinment_list  : list contains the info of matching.
    """
    
    return matched_weights, assignments_list


def BBP_MAP_frcnn(nets_list, 
                  model_meta_data, 
                  layer_type, 
                  #net_dataidx_map,
                  averaging_weights, 
#                   args,
                  traindata_cls_counts,
                  device="cpu"): 
    
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    n_classes = 10   ### args_net_config[-1]  ### BAD practice.
    it=5
    
    assignments_list = []
    
    ### prepare the weights of the existing models. (in fact, pdm_prepare_full_weights is in general form)
    batch_weights = pdm_prepare_full_weights_nchc(models, device=device)
    
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))
    
    ### calculate the freq. of classes.
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
#     res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    ### parameters for weights matching algorithm.
    gamma  = 7.0
    sigma  = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)  ### bad assumption?? assume all layers of the form [weights w/ bias].
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None
    
    print("*"*15 + "# layers: {:}".format(n_layers)) 
    
    ### matching layer by layer.
#     for layer_index in range(1, n_layers):
    for layer_index in range(1, n_layers):

#         layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent_full_cnn(
             batch_weights = batch_weights, 
             layer_index   = layer_index,
             sigma0_layers = sigma0, 
             sigma_layers  = sigma, 
             batch_frequencies = batch_freqs, 
             it=it, 
             gamma_layers     = gamma, 
             model_meta_data  = model_meta_data,
             model_layer_type = layer_type,
             n_layers = n_layers,
             matching_shapes = matching_shapes,
             args = None #args
             )
        
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2] 
        
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        ###  Determine the 1st fc layer ###########################
        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and 
                               ('conv' in type_of_prev_layer or 'features' in type_of_prev_layer))
        if first_fc_identifier:
            first_fc_index = layer_index
        ############################################################
        
        matching_shapes.append(L_next)
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) 
                          for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            
            
            if first_fc_index is None:
                if l_type == "conv":
                    ### [cyjui, 21/12/29]
#                     patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
#                                                     L_next, assignment[worker_index], 
#                                                     layer_index+1, model_meta_data,
#                                                     matching_shapes=matching_shapes, layer_type=l_type,
#                                                     dataset=args.dataset, 
#                                                     network_name=args.model)
                    patched_weight = block_patching_frcnn(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                                          L_next, assignment[worker_index], 
                                                          layer_index+1, model_meta_data,
                                                          matching_shapes=matching_shapes, 
                                                          layer_type=l_type,
                                                          network_name='vgg16'
                                                         )
                elif l_type == "fc":
                    patched_weight = block_patching_frcnn(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                                          L_next, assignment[worker_index], 
                                                          layer_index+1, model_meta_data,
                                                          matching_shapes=matching_shapes, 
                                                          layer_type=l_type,
                                                          network_name='vgg16'
                                                         ).T
#                 else:
#                     patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
#                                                     L_next, assignment[worker_index], 
#                                                     layer_index+1, model_meta_data,
#                                                     matching_shapes=matching_shapes, layer_type=l_type,
#                                                     dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                               L_next, assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])
        
        
        retrained_nets = []
        
        for worker_index in range(num_workers):
#             dataidxs = net_dataidx_map[worker_index]
#             train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)

            logger.info("Re-training on local worker: {}, starting from layer: {}".
                        format(worker_index, 2 * (layer_index + 1) - 2))
            
#             retrained_cnn = local_retrain_nchc((train_dl_local, test_dl_local), 
#                                                tempt_weights[worker_index], 
#                                                args, 
#                                                freezing_index=(2 * (layer_index + 1) - 2), device=device)
            
#             retrained_nets.append(retrained_cnn)
                    
#         batch_weights = pdm_prepare_full_weights_nchc(retrained_nets, device=device)
        batch_weights = tempt_weights
        
    ### ==========  end of layer wise matching & local retrain. ==============

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    ### save weights of all cleints.
    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    ### we don't handle the last layer particularly. cyjui, 21/12/30
#     last_layer_weights_collector = []

#     for i in range(num_workers):
#         # firstly we combine last layer's weight and bias
#         bias_shape = batch_weights[i][-1].shape
#         last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
#         last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
#         # the directed normalization doesn't work well, let's try weighted averaging
#         last_layer_weights_collector.append(last_layer_weights)

#     last_layer_weights_collector = np.array(last_layer_weights_collector)
    
#     avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

#     ### weighted average according to the population of classes.
#     for i in range(n_classes):
#         avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
#         for j in range(num_workers):
#             avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
#         avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    
    
    for i in range(num_layers):
#         if i < (num_layers - 2): ### don't have to deal the last layer. [cyjui, 21/12/30]
        if i < (num_layers):
            matched_weights.append(batch_weights[0][i])

#     matched_weights.append(avg_last_layer_weight[0:-1, :])
#     matched_weights.append(avg_last_layer_weight[-1, :])
    
    """
        matched_weights : identical to batch_weights(matched retrained model) except the last weights & bias.
        assinment_list  : list contains the info of matching.
    """
    
    return matched_weights, assignments_list


def freeze_VGG(vgg_features, layer_idx):
    """ Freeze VGG to certain layer.
    Args: vgg_features - vgg torch model.
          layer_index  - index number.
    """
    for layer in range(layer_idx):
        for p in vgg_features[layer].parameters(): p.requires_grad = False


def BBP_MAP_trivial(batch_weights,
                    assignments_list,
                    layer_index):
    """ return dummy results of BBP_MAP algorithm w/o touching the weights.
    Args: batch_weights - models' weights as list of numpy arrays.
          assignments_list - the permuation info of BBP_MAP
    Return: weights - temporla models' weights.
            asignments_list - updated assignments_list 
    """
    
    trivial_assignments = lambda weights: [i for i in range(len(weights))] 
    
    assignments = [trivial_assignments(batch_weight[2*layer_index - 2]) for idx, batch_weight in enumerate(batch_weights)]
    
    assignments_list.append(assignments)
    
    return batch_weights, assignments_list


def BBP_MAP_VGG(#nets_list,
                batch_weights,
                assignments_list,
                matching_shapes,
                layer_index,
#                 bbp_map_prms,
    
                model_meta_data, 
                layer_type,     ### "conv" for all cases in vgg version.
                #net_dataidx_map,
#                 averaging_weights, 
                #                   args,
                device="cpu"): 
    
    # starting the neural matching
    it = 5  ### number of iterations for lapsolver.
    
    ### prepare the weights of the existing models. (in fact, pdm_prepare_full_weights is in general form)
#     batch_weights = pdm_prepare_full_weights_nchc(models, device=device) ### cyjui, [22/01/03]
    
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))
     
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    ### parameters for weights matching algorithm.
    gamma  = 7.0
    sigma  = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)  ### bad assumption?? assume all layers of the form [weights w/ bias].
#     num_workers = len(nets_list)
    num_workers = len(batch_weights)
    
#     print("*"*15 + "# layers: {:}".format(n_layers))
    
    ### matching layer by layer.
#     for layer_index in range(1, n_layers):
    if True:

#         layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent_full_cnn(
             batch_weights = batch_weights, 
             layer_index   = layer_index,
             sigma0_layers = sigma0, #bbp_map_prms["sigma0"], 
             sigma_layers  = sigma, #bbp_map_prms["sigma"], 
#              batch_frequencies = batch_freqs, 
             it=it, 
             gamma_layers     = gamma, #bbp_map_prms["gamma"], 
             model_meta_data  = model_meta_data,
             model_layer_type = layer_type,
             n_layers = n_layers,
#              matching_shapes = matching_shapes,
             args = None #args
             )
        
        assignments_list.append(assignment)
        
        print("="*15 + "assignment: {:}, weights: {:}".format(len(assignment[0]), layer_hungarian_weights[0][layer_index].shape))
        
        ### we care about conv layer only.
        # iii) load weights to the model and train the whole thing
#         type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2] 
        
#         if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
#             l_type = "conv"
            
        l_type = "conv"
            
        ############################################################
        
        matching_shapes.append(L_next)
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) 
                          for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        if layer_index < n_layers:
            
            for worker_index in range(num_workers):

                if l_type == "conv":
                    ### [cyjui, 21/12/29]
    #                 print("{:}, {:}, {:}".format(len(batch_weights[worker_index]), layer_index, len(assignment)))

                    patched_weight = block_patching_frcnn(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                                          L_next, assignment[worker_index], 
                                                          layer_index+1, model_meta_data,
                                                          matching_shapes=matching_shapes, 
                                                          layer_type=l_type,
                                                          network_name='vgg16'
                                                         )

                tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights ### layers after the matched weights.
        for worker_index in range(num_workers):
#             for lid in range(2 * (layer_index) - 1, len(batch_weights[0])): ### note that we don't deal the last layer particularly.
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])
        
        
        return tempt_weights, assignments_list
####### End here.        
        

        ### here the weights are completed.
        
        retrained_nets = []
        
        for worker_index in range(num_workers):
#             dataidxs = net_dataidx_map[worker_index]
#             train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)

            logger.info("Re-training on local worker: {}, starting from layer: {}".
                        format(worker_index, 2 * (layer_index + 1) - 2))
            
#             retrained_cnn = local_retrain_nchc((train_dl_local, test_dl_local), 
#                                                tempt_weights[worker_index], 
#                                                args, 
#                                                freezing_index=(2 * (layer_index + 1) - 2), device=device)
            
#             retrained_nets.append(retrained_cnn)
                    
#         batch_weights = pdm_prepare_full_weights_nchc(retrained_nets, device=device)
#         batch_weights = tempt_weights
        
    ### ==========  end of layer wise matching & local retrain. ==============

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

#     ### save weights of all cleints.
#     with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
#         pickle.dump(batch_weights, weights_file)

    ### we don't handle the last layer particularly. cyjui, 21/12/30
    
#     last_layer_weights_collector = []

#     for i in range(num_workers):
#         # firstly we combine last layer's weight and bias
#         bias_shape = batch_weights[i][-1].shape
#         last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
#         last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
#         # the directed normalization doesn't work well, let's try weighted averaging
#         last_layer_weights_collector.append(last_layer_weights)

#     last_layer_weights_collector = np.array(last_layer_weights_collector)
    
#     avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

#     ### weighted average according to the population of classes.
#     for i in range(n_classes):
#         avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
#         for j in range(num_workers):
#             avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
#         avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    
    
    for i in range(num_layers):
#         if i < (num_layers - 2): ### don't have to deal the last layer specifically. [cyjui, 21/12/30]
        if i < (num_layers):
            matched_weights.append(batch_weights[0][i]) ### padding

#     matched_weights.append(avg_last_layer_weight[0:-1, :])
#     matched_weights.append(avg_last_layer_weight[-1, :])
    
    """
        matched_weights : identical to batch_weights(matched retrained model) except the last weights & bias.
        assinment_list  : list contains the info of matching.
    """
    
    return matched_weights, assignments_list



def BBP_MAP_FC(#nets_list,
                batch_weights,
                assignments_list,
                matching_shapes,
                layer_index,
#                 bbp_map_prms,
    
                model_meta_data, 
                layer_type='fc',     ### "conv" for all cases in vgg version.
                #net_dataidx_map,
#                 averaging_weights, 
                #                   args,
                device="cpu"): 
    
    # starting the neural matching
    it = 5  ### number of iterations for lapsolver.
    
    ### prepare the weights of the existing models. (in fact, pdm_prepare_full_weights is in general form)
#     batch_weights = pdm_prepare_full_weights_nchc(models, device=device) ### cyjui, [22/01/03]
    
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))
     
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    ### parameters for weights matching algorithm.
    gamma  = 7.0
    sigma  = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)  ### bad assumption?? assume all layers of the form [weights w/ bias].
#     num_workers = len(nets_list)
    num_workers = len(batch_weights)
    
#     print("*"*15 + "# layers: {:}".format(n_layers))
    
    ### matching layer by layer.
    if True:

#         layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
#         layer_hungarian_weights, assignment, L_next = layer_wise_group_descent_full_cnn(
#              batch_weights = batch_weights, 
#              layer_index   = layer_index,
#              sigma0_layers = sigma0, #bbp_map_prms["sigma0"], 
#              sigma_layers  = sigma, #bbp_map_prms["sigma"], 
# #              batch_frequencies = batch_freqs, 
#              gamma_layers     = gamma, #bbp_map_prms["gamma"], 
#              it=it, 
#              model_meta_data  = model_meta_data,
#              model_layer_type = layer_type,
#              n_layers = n_layers,
# #              matching_shapes = matching_shapes,
#              #args
#              )
    
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
            batch_weights = batch_weights, 
            layer_index   = layer_index,
            batch_frequencies= [1, 1, 1, 1], 
            sigma0_layers = sigma0, #bbp_map_prms["sigma0"], 
            sigma_layers  = sigma, #bbp_map_prms["sigma"], 
            gamma_layers  = gamma, #bbp_map_prms["gamma"], 
            it=it,
            model_meta_data = model_meta_data, 
            model_layer_type = layer_type,
            n_layers = n_layers,
            matching_shapes = None,
            args = None)
        
        assignments_list.append(assignment)
        
        print("="*15 + "assignment: {:}, weights: {:}".format(len(assignment[0]), layer_hungarian_weights[0][layer_index].shape))
        
        ### we care about conv layer only.
        # iii) load weights to the model and train the whole thing
#         type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2] 
        
#         if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
#             l_type = "conv"
            
        l_type = "fc"
            
        ############################################################
        
        matching_shapes.append(L_next)
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) 
                          for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            
            if l_type == "conv":
                ### [cyjui, 21/12/29]
#                 print("{:}, {:}, {:}".format(len(batch_weights[worker_index]), layer_index, len(assignment)))
                
                patched_weight = block_patching_frcnn(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                                      L_next, assignment[worker_index], 
                                                      layer_index+1, model_meta_data,
                                                      matching_shapes=matching_shapes, 
                                                      layer_type=l_type,
                                                      network_name='vgg16'
                                                     )

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights ### layers after the matched weights.
        for worker_index in range(num_workers):
#             for lid in range(2 * (layer_index) - 1, len(batch_weights[0])): ### note that we don't deal the last layer particularly.
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])
        
        
        return tempt_weights, assignments_list



def reconstruct_local_net_frcnn(weights, 
#                                 args, 
                                ori_assignments=None, 
                                worker_index=0):
    
    def __reconstruct_weights(weight, assignment, 
                              layer_ori_shape, 
                              matched_num_filters=None, 
                              weight_type="conv_weight", slice_dim="filter"):
        
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight  = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight    = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
                
        elif weight_type == "bias":
            res_weight = weight[assignment]
            
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :]
                
        return res_weight
    
    ################################################
    
    ### select base model type, just use to estimate the output shape.
    if True: #args.model == "vgg":
        
        ### Original net arch.
        matched_cnn = vgg16_rcnn()
        
        shape_estimator = matched_cnn.features

        dummy_input = torch.rand(1, 3, 224, 224)

        estimated_output = shape_estimator(dummy_input)
        
        print("Estimated output shape: {:}".format(estimated_output.shape))
       
    print("##### In reconsturct_local_net[assignments] #####")
    for idx, a in enumerate(ori_assignments):
        print(idx, len(a[0]))
    
    print("##### In reconsturct_local_net #####")
    for idx, w in enumerate(weights):
        print(idx, w.shape)
    
    print("##### matched model #####")
    print(matched_cnn)

    reconstructed_weights = []
    
    new_state_dict = {}
    
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        
        _matched_weight = weights[param_idx]  ### This is from global model, it might larger than the local arch.
        
        print("layer: {:}, matched_w: {:}, cnn_w: {:}".format(param_idx, param.size(), _matched_weight.shape))
        
        ### dealing with input layer.
        if param_idx < 1: # we need to handle the 1st conv layer specificly since the color channels are aligned
            _assignment = ori_assignments[int(param_idx / 2)][worker_index]
            _res_weight = __reconstruct_weights(weight=_matched_weight, 
                                                assignment=_assignment, 
                                                layer_ori_shape=param.size(), 
                                                matched_num_filters=None,
                                                weight_type="conv_weight", slice_dim="filter")
            
            reconstructed_weights.append(_res_weight)

        ### all layers except the first.
        elif (param_idx >= 1) and (param_idx < len(weights)-2): ### cyjui, 21/12/30.
#         elif (param_idx >= 1) and (param_idx < len(weights) -2):
            if "bias" in key_name: # the last bias layer is already aligned so we won't need to process it
                _assignment = ori_assignments[int(param_idx / 2)][worker_index]
                _res_bias = __reconstruct_weights(weight=_matched_weight, 
                                                  assignment=_assignment, 
                                                  layer_ori_shape=param.size(), 
                                                  matched_num_filters=None,
                                                  weight_type="bias", 
                                                  slice_dim=None)
                
                reconstructed_weights.append(_res_bias)

#             elif "conv" in key_name or "features" in key_name:
            elif "weight" in key_name or "features" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment  = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                
                _matched_num_filters = weights[param_idx - 2].shape[0]
                _layer_ori_shape = list(param.size())
                _layer_ori_shape[0] = _matched_weight.shape[0]
                
                print("param_idx: {:}, _matched_num_filters: {:}".format(param_idx, _matched_num_filters))

                _temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                    layer_ori_shape=_layer_ori_shape, matched_num_filters=_matched_num_filters,
                                                    weight_type="conv_weight", slice_dim="channel")

                _res_weight = __reconstruct_weights(weight=_temp_res_weight, assignment=cur_assignment, 
                                                    layer_ori_shape=param.size(), matched_num_filters=None,
                                                    weight_type="conv_weight", slice_dim="filter")
                
                reconstructed_weights.append(_res_weight)
            else:
                print("param_idx: {:}, {:}, ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(param_idx, key_name))

### we don't deal the fc layers.
#             elif "fc" in key_name or "classifier" in key_name:
#                 # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
#                 cur_assignment = ori_assignments[int(param_idx / 2)][worker_index]
#                 prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
#                 _matched_num_filters = weights[param_idx - 2].shape[0]

#                 if param_idx != 16: ### cyjui for vgg [21/12/15]
# #                 if param_idx != 12: # this is the index of the first fc layer
#                     #logger.info("%%%%%%%%%%%%%%% prev assignment length: {}, 
#                     #cur assignmnet length: {}".format(len(prev_assignment), len(cur_assignment)))
#                     temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="fc_weight", slice_dim="channel")

#                     _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=None,
#                                                         weight_type="fc_weight", slice_dim="filter")

#                     reconstructed_weights.append(_res_weight.T)
#                 else:
#                     # that's for handling the first fc layer that is connected to the conv blocks
#                     temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment,
#                                                             layer_ori_shape=estimated_output.size(), 
#                                                             matched_num_filters=_matched_num_filters,
#                                                             weight_type="first_fc_weight", slice_dim=None)

#                     _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=None,
#                                                         weight_type="fc_weight", slice_dim="filter")
#                     reconstructed_weights.append(_res_weight.T)
           
#### identical treatment for last layer.
        elif param_idx == len(weights) - 2:  ### We cheat for the last layer.
#             # this is to handle the weight of the last layer
#             prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
# #             prev_assignment = 512 ### we cheat for the last layer.
# #             print("prev_assignment = {:}, _matched_weight shape{:}".format(len(prev_assignment), 
# #                                                                           _matched_weight.shape))
#             _res_weight = _matched_weight[prev_assignment, :] ### original 
#             _res_weight = _matched_weight[prev_assignment, :512]  ### incorrect cheating.
            
#             _res_weight = _matched_weight.reshape((512, 3, 3))
            reconstructed_weights.append(_res_weight)
            
        elif param_idx == len(weights) - 1:
#             print("last bias shape: {:}".format(_matched_weight.shape))
#             reconstructed_weights.append(_matched_weight)
            ### pass forward the last bias.
            reconstructed_weights.append(_matched_weight)

    return reconstructed_weights


def prepare_for_local_retrain(#local_datasets, 
                              weights, 
#                               args, 
                              mode="bottom-up", 
                              freezing_index=0, 
                              ori_assignments=None, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freeze the entire network
    """
    ###############################################################################   
    def __reconstruct_weights(weight, assignment, 
                              layer_ori_shape, matched_num_filters=None, 
                              weight_type="conv_weight", slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the 
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to 
               slice the color channel 
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight  = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
            
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
            
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
                
        return res_weight
############################################################################   
    
    ### preapare the base model.
    if True:
        
        ### allocate VGG net according to the shapes of weights. 
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg16_no_FC(matched_shapes=matched_shapes)

    
    new_state_dict = {}
    
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    #if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"): ### mode = "bottom-up" for default configuration, always this way?
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(__unfreezing_list[-1]+1) # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]
 
 
    ### build model with specific shape according to the weights.
    matched_shapes = [w.shape for w in weights]
#     matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
        

    ### the first FC layer size of matched_cnn is INCORRECT. 
    
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        
        ### This won't be execute since default __non_loading_indices = []
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            pass
#             # we need to reconstruct the weights here s.t.
#             # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
#             # ii) each neuron comes from the corresponding global neuron
#             _matched_weight = weights[param_idx]
#             _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
#             #
#             # we now use this `_slice_dim` for both conv layers and fc layers
#             if __non_loading_indices.index(param_idx) != 2:
#                 _slice_dim = "filter" # please note that for biases, it doesn't really matter if we're going to use filter or channel
#             else:
#                 _slice_dim = "channel"

#             if "conv" in key_name or "features" in key_name:
#                 if "weight" in key_name:
#                     _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="conv_weight", slice_dim=_slice_dim)
#                     temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
#                 elif "bias" in key_name:
#                     _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="bias", slice_dim=_slice_dim)                   
#                     temp_dict = {key_name: torch.from_numpy(_res_bias)}
                    
#             elif "fc" in key_name or "classifier" in key_name:
#                 if "weight" in key_name:
#                     if freezing_index[0] != 6:
#                         _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                             layer_ori_shape=param.size(), matched_num_filters = _matched_num_filters,
#                                                             weight_type="fc_weight", slice_dim=_slice_dim)
#                         temp_dict = {key_name: torch.from_numpy(_res_weight)}
#                     else:
#                         # that's for handling the first fc layer that is connected to the conv blocks
#                         _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                             layer_ori_shape = estimated_output.size(), 
#                                                             matched_num_filters = _matched_num_filters,
#                                                             weight_type="first_fc_weight", slice_dim=_slice_dim)
#                         temp_dict = {key_name: torch.from_numpy(_res_weight.T)}    
                        
#                 elif "bias" in key_name:
#                     _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="bias", slice_dim=_slice_dim)
#                     temp_dict = {key_name: torch.from_numpy(_res_bias)}

        else: ### always this branch.
            
            if True: #"conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                    
#             elif "fc" in key_name or "classifier" in key_name:
                
#                 if "weight" in key_name:
#                     temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
#                 elif "bias" in key_name:
#                     temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
        
    
    ### Show the shapes of the dictionary.
    for param_idx, (key_name, param) in enumerate(new_state_dict.items()):
        print("param_idx {:}: {:}, {:}".format(param_idx, key_name, param.shape))
        
    
    matched_cnn.load_state_dict(new_state_dict)
    
    return matched_cnn
    


def local_retrain_nchc(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freeze the entire network
    """
    ###############################################################################   
    def __reconstruct_weights(weight, assignment, 
                              layer_ori_shape, matched_num_filters=None, 
                              weight_type="conv_weight", slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the 
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to 
               slice the color channel 
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight  = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
            
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
            
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
                
        return res_weight
############################################################################   
    
    ### preapare the base model.
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".
                    format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
        
    elif args.model == "vgg":
        
        ### allocate VGG net according to the shapes of weights. 
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)

    
    new_state_dict = {}
    
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    #if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"): ### mode = "bottom-up" for default configuration, always this way?
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(__unfreezing_list[-1]+1) # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]
 
 
    ### build model with specific shape according to the weights.
    matched_shapes = [w.shape for w in weights]
    matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
        

    ### the first FC layer size of matched_cnn is INCORRECT. 
    
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        
        ### This won't be execute since default __non_loading_indices = []
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            pass
#             # we need to reconstruct the weights here s.t.
#             # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
#             # ii) each neuron comes from the corresponding global neuron
#             _matched_weight = weights[param_idx]
#             _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
#             #
#             # we now use this `_slice_dim` for both conv layers and fc layers
#             if __non_loading_indices.index(param_idx) != 2:
#                 _slice_dim = "filter" # please note that for biases, it doesn't really matter if we're going to use filter or channel
#             else:
#                 _slice_dim = "channel"

#             if "conv" in key_name or "features" in key_name:
#                 if "weight" in key_name:
#                     _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="conv_weight", slice_dim=_slice_dim)
#                     temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
#                 elif "bias" in key_name:
#                     _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="bias", slice_dim=_slice_dim)                   
#                     temp_dict = {key_name: torch.from_numpy(_res_bias)}
                    
#             elif "fc" in key_name or "classifier" in key_name:
#                 if "weight" in key_name:
#                     if freezing_index[0] != 6:
#                         _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                             layer_ori_shape=param.size(), matched_num_filters = _matched_num_filters,
#                                                             weight_type="fc_weight", slice_dim=_slice_dim)
#                         temp_dict = {key_name: torch.from_numpy(_res_weight)}
#                     else:
#                         # that's for handling the first fc layer that is connected to the conv blocks
#                         _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                             layer_ori_shape = estimated_output.size(), 
#                                                             matched_num_filters = _matched_num_filters,
#                                                             weight_type="first_fc_weight", slice_dim=_slice_dim)
#                         temp_dict = {key_name: torch.from_numpy(_res_weight.T)}    
                        
#                 elif "bias" in key_name:
#                     _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
#                                                         layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
#                                                         weight_type="bias", slice_dim=_slice_dim)
#                     temp_dict = {key_name: torch.from_numpy(_res_bias)}

        else: ### always this branch.
            
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                    
            elif "fc" in key_name or "classifier" in key_name:
                
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
        
    
    matched_cnn.load_state_dict(new_state_dict)
    
    ### Determine if there are layers to freeze.
    for param_idx, param in enumerate(matched_cnn.parameters()):
        if mode == "bottom-up":
            # for this freezing mode, we freeze the layer before freezing index
            if param_idx < freezing_index:
                param.requires_grad = False
        elif mode == "per-layer":
            # for this freezing mode, we only unfreeze the freezing index
            if param_idx not in (2*freezing_index-2, 2*freezing_index-1):
                param.requires_grad = False
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            if param_idx not in __non_loading_indices:
                param.requires_grad = False
        elif mode == "squeezing":
            pass

    
    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local  = local_datasets[1]

    if mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), 
                                            lr=args.retrain_lr, momentum=0.9)
        else:
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), 
                                            lr=(args.retrain_lr/10), momentum=0.9, weight_decay=0.0001)
    else:
        #optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9)
        optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), 
                                         lr=0.001, weight_decay=0.0001, amsgrad=True)
    
    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: %d' % len(train_dl_local))
    logger.info('n_test: %d' % len(test_dl_local))

    train_acc, test_acc = 0, 0 ### cyjui
    
#     train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)
    

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    ### This is tricky? Is it fair to use different # of epochs???
    if mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            retrain_epochs = args.retrain_epochs
        else:
            retrain_epochs = int(args.retrain_epochs*3)
    else:
        retrain_epochs = args.retrain_epochs

    for epoch in range(retrain_epochs):
        
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dl_local):
            
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)
            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()
            
            ### cyjui.
            NUM2SKIP = 1
            if  batch_idx > NUM2SKIP:
                print("@"*15 + "Break after {:} single batch".format(NUM2SKIP) + "@"*15)
                break

        #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc, test_acc = 0, 0 # cyjui.
#     train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    
    return matched_cnn
