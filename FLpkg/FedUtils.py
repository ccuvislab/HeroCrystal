def avgWeight(model_list,ratio_list,keywords=None):
    parties = len(model_list)
    model_tmp=[None] * parties
    #optims_tmp=[None] * parties

    for idx, my_model in enumerate(model_list):
        print(next(my_model.parameters()).device)
        
        model_tmp[idx] = my_model.state_dict()


    for key in model_tmp[0]:    
        if (keywords is None) or (keywords in key):
            #print(key)
            model_avg = 0

            for idx, model_tmp_content in enumerate(model_tmp):     # add each model              
                model_avg += ratio_list[idx] * model_tmp_content[key]
                
            for i in range(len(model_tmp)):  #copy to each model            
                model_tmp[i][key] = model_avg
    for i in range(len(model_list)):    
        model_list[i].load_state_dict(model_tmp[i])
        
    return model_list  #, optims_tmp
    
    
    
def avg_model_weight(model_list):
    balance = 1/len(model_list)
    ratio_list= [balance] * len(model_list)
    
    avg_model=avgWeight(model_list,ratio_list)
    return avg_model


def get_backbone_shape(fedma_model):
    backbone_shape = []
    for idx,(key,value) in enumerate(fedma_model.items()):
        if idx%2!=0 and idx<26:
            backbone_shape.append(list(value.shape)[0])
    return backbone_shape