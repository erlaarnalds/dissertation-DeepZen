from obj_models.datasets import IceandFire_SA, IceandFire_hate, IceandFire_offense, \
                                RU, REST14, CompSent19, REST16_UABSA, IceandFire_Irony, \
                                Stance, Implicit, IceandFire_ER, ASTE, ASQP

def get_dataset_obj(dataset_name, dataset_path, shot):
    if dataset_name == "ice_and_fire_SA":
        return IceandFire_SA(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)
    
    elif dataset_name == "RU_imdb_google" or dataset_name == "RU_movie_reviews":
        return RU(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)

    elif dataset_name == "asc_rest14":
        return REST14(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)

    if dataset_name == "ice_and_fire_hate":
        return IceandFire_hate(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)
    
    if dataset_name == "ice_and_fire_offensive":
        return IceandFire_offense(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)

    if dataset_name == "ice_and_fire_irony":
        return IceandFire_Irony(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)

    if dataset_name == "ice_and_fire_ER":
        return IceandFire_ER(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)

    if dataset_name == "compsent19":
        return CompSent19(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)
    
    if dataset_name == "uabsa_rest16":
        return REST16_UABSA(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)
    
    if dataset_name == "stance":
        return Stance(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)

    if dataset_name == "implicit":
        return Implicit(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)
    
    if dataset_name == "aste_rest14":
        return ASTE(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)
    
    if dataset_name == "asqp_rest15":
        return ASQP(dataset_path=dataset_path, dataset_name=dataset_name, shot=shot)


def process_tuple_f1(labels_str, predictions_str, verbose=False):
    tp, fp, fn, tn = 0, 0, 0, 0
    epsilon = 1e-7
    for i in range(len(labels_str)):
        if type(labels_str[i]) == str:
            label = str_to_tuple(labels_str[i])
        else:
            label = labels_str[i]
        gold = set(label)
        try:
            if type(predictions_str[i]) == str:
                prediction = str_to_tuple(predictions_str[i])
            else:
                prediction = predictions_str[i]
            pred = set(prediction)
        except Exception:
            pred = set()
        tp += len(gold.intersection(pred))
        fp += len(pred.difference(gold))
        fn += len(gold.difference(pred))

        if verbose:
            print('-'*100)
            print(gold)
            print(pred)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    micro_f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision, recall, micro_f1