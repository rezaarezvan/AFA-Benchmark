import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
from sklearn import preprocessing
from sklearn.utils import shuffle
from afa_generative.afa_methods import (
    EDDI,
    UniformSampler,
    IterativeSelector,
    SGHMC,
    base_Active_Learning_SGHMC_Decoder,
)
from afa_generative.utils import (
    ReadYAML,
    test_UCI_AL,
    Compute_AUIC_1D,
    remove_zero_row_2D,
    MaskLayer,
)
from afa_generative.models import PVAE, Point_Net_Plus_BNN_SGHMC, fc_Net
from afa_generative.datasets import (
    load_spam,
    load_diabetes,
    load_miniboone,
    data_split,
    get_xy,
    Boston,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="spam",
    choices=["spam", "diabetes", "miniboone", "boston"],
)
parser.add_argument(
    "--method", type=str, default="eddi", choices=["eddi", "Icebreaker"]
)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--num_restarts", type=int, default=1)


# Various configurations.
load_data_dict = {
    "spam": load_spam,
    "diabetes": load_diabetes,
    "miniboone": load_miniboone,
}
num_features_dict = {
    "spam": list(range(1, 11)) + list(range(15, 30, 5)),
    "diabetes": list(range(1, 11)),
    "miniboone": list(range(1, 11)) + list(range(15, 30, 5)),
}
max_features_dict = {
    "spam": 35,
    "diabetes": 35,
    "miniboone": 35,
}

cfg = ReadYAML("./configs/afa_generative/Config_SGHMC_PNP_UCI0.yaml")
max_selection_eval = cfg["AL_Eval_Settings"][
    "max_selection"
]  # This for Active learning of evaluation
encoder_layer_num_before_agg = cfg["BNN_Settings"]["encoder_settings"][
    "encoder_layer_num_before_agg"
]
encoder_hidden_before_agg = cfg["BNN_Settings"]["encoder_settings"][
    "encoder_hidden_before_agg"
]
encoder_layer_num_after_agg = cfg["BNN_Settings"]["encoder_settings"][
    "encoder_layer_num_after_agg"
]
encoder_hidden_after_agg = cfg["BNN_Settings"]["encoder_settings"][
    "encoder_hidden_after_agg"
]
decoder_layer_num = cfg["BNN_Settings"]["decoder_settings"]["decoder_layer_num"]
decoder_hidden = cfg["BNN_Settings"]["decoder_settings"]["decoder_hidden"]
pooling = cfg["BNN_Settings"]["encoder_settings"]["pooling"]
output_const = cfg["BNN_Settings"]["decoder_settings"]["output_const"]
sample_z = cfg["BNN_Settings"]["encoder_settings"]["sample_z"]
sample_W = cfg["BNN_Settings"]["decoder_settings"]["sample_W"]
sample_W_PNP = 1
pooling_act = cfg["BNN_Settings"]["encoder_settings"]["pooling_act"]
BNN_init_range = cfg["BNN_Settings"]["decoder_settings"]["init_range"]
BNN_coef_sample = cfg["BNN_Settings"]["decoder_settings"]["coef_sample"]
KL_coef = cfg["BNN_Settings"]["KL_coef"]
W_sigma_prior = cfg["Training_Settings"]["W_sigma_prior"]
sigma_out = 0.4
latent_dim = cfg["BNN_Settings"]["latent_dim"]
dim_before_agg = cfg["BNN_Settings"]["dim_before_agg"]
embedding_dim = cfg["BNN_Settings"]["embedding_dim"]
add_const = cfg["BNN_Settings"]["decoder_settings"]["add_const"]
flag_log_q = cfg["BNN_Settings"]["flag_log_q"]

flag_clear_target_train = cfg["Active_Learning_Settings"]["flag_clear_target_train"]
flag_clear_target_test = cfg["Active_Learning_Settings"]["flag_clear_target_test"]

flag_hybrid = cfg["Active_Learning_Settings"]["flag_hybrid"]
conditional_coef = 0.8
conditional_coef_sghmc = 0.8
balance_coef = cfg["Active_Learning_Settings"]["balance_coef"]
BALD_coef = 0.5
# KL Schedule
KL_Schedule_Settings = cfg["KL_Schedule_Settings"]
KL_coef_W = None
# KL Pretrain_Schedule
KL_Schedule_Pretrain_Settings = cfg["KL_Schedule_Pretrain_Settings"]
KL_coef_W_pretrain = None

max_selection = cfg["Active_Learning_Settings"][
    "max_selection"
]  # This for active learning of W
flag_pretrain = cfg["Pretrain_Settings"]["flag_pretrain"]
step = cfg["Active_Learning_Settings"]["step"]

flag_reset_optim = cfg["Training_Settings"]["flag_reset_optim"]
Drop_orig = cfg["Training_Settings"]["Drop_p"]
cfg["Optimizer"]["lr_sghmc"] = 0.003
epoch_orig = 1500
tot_epoch = 1500
step_sghmc = 4e-4
scale_data = 1
noisy_update = 0
selection_scheme = "overall"


if __name__ == "__main__":
    # Parse args.
    args = parser.parse_args()
    device = torch.device("cuda", args.gpu)

    # Load, normalize, and split dataset.
    if args.method == "eddi":
        load_data = load_data_dict[args.dataset]
        num_features = num_features_dict[args.dataset]
        dataset = load_data()
        d_in = dataset.input_size
        d_out = dataset.output_size
        mean = dataset.tensors[0].mean(dim=0)
        std = torch.clamp(dataset.tensors[0].std(dim=0), min=1e-3)
        # PVAE generative model works better with standardized data.
        dataset.tensors = ((dataset.tensors[0] - mean) / std, dataset.tensors[1])
        train_dataset, val_dataset, test_dataset = data_split(
            dataset, val_portion=0.2, test_portion=0.2
        )
        # Prepare dataloaders.
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=1024, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, pin_memory=True)
    elif args.method == "Icebreaker":
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)
        if args.dataset == "boston":
            dataset = Boston("./data/boston")
            obs_dim = dataset.Data_mat.shape[1]

    # Make results directory.
    # if not os.path.exists('results'):
    #     os.makedirs('results')

    for trial in range(args.num_trials):
        # For saving results.
        # results_dict = {
        #     'auroc': {},
        #     'acc': {},
        #     'features': {}
        # }
        # auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
        # acc_metric = Accuracy(task='multiclass', num_classes=d_out)

        if args.method == "eddi":
            # Train PVAE.
            bottleneck = 16
            encoder = fc_Net(
                input_dim=d_in * 2,
                output_dim=bottleneck * 2,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations="ReLU",
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False,
            )
            decoder = fc_Net(
                input_dim=bottleneck,
                output_dim=d_in,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations="ReLU",
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False,
            )
            mask_layer = MaskLayer(append=True)
            pv = PVAE(encoder, decoder, mask_layer, 128, "gaussian").to(device)
            pv.fit(train_loader, val_loader, lr=1e-3, nepochs=250, verbose=True)

            # Train masked predictor.
            model = fc_Net(
                input_dim=d_in * 2,
                output_dim=d_out,
                hidden_layer_num=2,
                hidden_unit=[128, 128],
                activations="ReLU",
                drop_out_rate=0.3,
                flag_drop_out=True,
                flag_only_output_layer=False,
            )
            sampler = UniformSampler(
                get_xy(train_dataset)[0]
            )  # TODO don't actually need sampler
            iterative = IterativeSelector(model, mask_layer, sampler).to(device)
            iterative.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True,
            )

            # Set up EDDI feature selection object.
            eddi_selector = EDDI(pv, model, mask_layer, "classification").to(device)

            # # Evaluate
            # metrics_dict = eddi_selector.evaluate_multiple(test_loader, num_features, (auroc_metric, acc_metric))
            # for num in num_features:
            #     auroc, acc = metrics_dict[num]
            #     results_dict['auroc'][num] = auroc
            #     results_dict['acc'][num] = acc
            #     print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')

        elif args.method == "Icebreaker":
            counter_loop = 0
            counter_selection = 0
            PNP_SGHMC = Point_Net_Plus_BNN_SGHMC(
                latent_dim=cfg["BNN_Settings"]["latent_dim"],
                obs_dim=obs_dim,
                dim_before_agg=cfg["BNN_Settings"]["dim_before_agg"],
                encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                encoder_hidden_before_agg=encoder_hidden_before_agg,
                encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                encoder_hidden_after_agg=encoder_hidden_after_agg,
                embedding_dim=embedding_dim,
                decoder_layer_num=decoder_layer_num,
                decoder_hidden=decoder_hidden,
                pooling=pooling,
                output_const=output_const,
                add_const=add_const,
                sample_z=sample_z,
                sample_W=sample_W,
                W_sigma_prior=W_sigma_prior,
                pooling_act=pooling_act,
                flag_log_q=flag_log_q,
            )

            Infer_SGHMC = SGHMC(model=PNP_SGHMC, Infer_name="Scale Adapted SGHMC")
            # Parameter List
            list_p_z = (
                list(PNP_SGHMC.encoder_before_agg.parameters())
                + list(PNP_SGHMC.encoder_after_agg.parameters())
                + [PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias]
            )

            Adam_encoder = torch.optim.Adam(
                list(PNP_SGHMC.encoder_before_agg.parameters())
                + list(PNP_SGHMC.encoder_after_agg.parameters()),
                lr=cfg["Optimizer"]["lr_sghmc"],
                betas=(cfg["Optimizer"]["beta1"], cfg["Optimizer"]["beta2"]),
                weight_decay=cfg["Optimizer"]["weight_decay"],
            )
            Adam_embedding = torch.optim.Adam(
                [PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias],
                lr=cfg["Optimizer"]["lr_sghmc"],
                betas=(cfg["Optimizer"]["beta1"], cfg["Optimizer"]["beta2"]),
                weight_decay=cfg["Optimizer"]["weight_decay"],
            )

            num_selected_variable = cfg["Active_Learning_Settings"]["step"]
            Dict_training_settings = cfg["Training_Settings"]
            Dict_dataset_settings = cfg["Dataset_Settings"]
            Drop_p = 0.6

            Active_BALD = base_Active_Learning_SGHMC_Decoder(
                model=PNP_SGHMC,
                Infer_model=Infer_SGHMC,
                overall_data=dataset.Data_mat,
                rs=42,
                sigma_out=sigma_out,
                Optim_settings=cfg["Optimizer"],
                Adam_encoder=Adam_encoder,
                Adam_embedding=Adam_embedding,
                flag_clear_target_train=flag_clear_target_train,
                flag_clear_target_test=flag_clear_target_test,
                model_name="SGHMC Active",
            )

            Active_BALD._data_preprocess(**Dict_dataset_settings)
            Active_BALD._get_pretrain_data(
                pretrain_number=cfg["Pretrain_Settings"]["pretrain_number"]
            )
            pretrain_data = Active_BALD.pretrain_data_tensor.clone().detach()
            valid_data_tensor = Active_BALD.valid_data_tensor.clone().detach()
            valid_data_input_tensor = None
            valid_data_target_tensor = (
                Active_BALD.valid_data_target_tensor.clone().detach()
            )
            # train_full_pool = torch.tensor(Active_BALD.train_data_tensor.data)
            train_full_pool = Active_BALD.train_data_tensor.clone().detach()

            # Pretrain
            W_sample = Active_BALD.train_BNN(
                pretrain_data,
                eps=step_sghmc,
                max_sample_size=40,
                tot_epoch=tot_epoch + 500,
                thinning=10,
                hyper_param_update=25000,
                sample_int=10,
                flag_hybrid=flag_hybrid,
                Adam_encoder=Adam_encoder,
                Adam_embedding=Adam_embedding,
                Drop_p=Drop_p,
                list_p_z=list_p_z,
                test_input_tensor=Active_BALD.test_input_tensor,
                test_target_tensor=Active_BALD.test_target_tensor,
                conditional_coef=conditional_coef_sghmc,
                target_dim=-1,
                flag_reset_optim=flag_reset_optim,
                valid_data=valid_data_input_tensor,
                valid_data_target=valid_data_target_tensor,
                sigma_out=sigma_out,
                scale_data=scale_data,
                noisy_update=noisy_update,
            )

            # Initialized samples
            W_dict_init = None
            # Prepare the training data
            train_pool_data_BALD = Active_BALD.train_pool_tensor.clone().detach()
            train_full_pool = Active_BALD.train_data_tensor.clone().detach()
            observed_train_BALD = (
                Active_BALD.init_observed_train_tensor.clone().detach()
            )

            train_data_perm = Active_BALD.train_data_tensor.clone().detach()
            # Initialization for test data
            test_pool_perm = Active_BALD.test_input_tensor.clone().detach()
            test_input_perm = torch.zeros(test_pool_perm.shape)
            test_pool_data_BALD_BALD = torch.tensor(Active_BALD.test_input_tensor.data)

            test_target_data = Active_BALD.test_target_tensor.clone().detach()

            # Initialize test input
            test_input_BALD_BALD = torch.zeros(Active_BALD.test_input_tensor.shape)

            # RMSE_BALD_BALD, MAE_BALD_BALD, NLL_BALD_BALD = test_UCI_AL(
            #     model=Active_BALD, max_selection=max_selection_eval,sample_x=None,
            #     test_input=test_input_BALD_BALD,test_pool=test_pool_data_BALD_BALD,
            #     test_target=test_target_data, sigma_out=sigma_out,search='Target',
            #     model_name='PNP_SGHMC', W_sample=W_sample)

            # AUIC_BALD_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_BALD_BALD)
            # Results_list = ['%s Init' % (0 * step), '%.3f' % (AUIC_BALD_BALD)]
            # print(Results_list)

            counter_loop += 1
            idx_start = 0
            idx_end = 0
            while counter_selection < max_selection:
                if counter_loop == 1:
                    flag_init_train = True
                else:
                    flag_init_train = False

                # Initialization for test data
                test_pool_data_BALD_BALD = test_pool_perm.clone().detach()
                test_target_data = test_target_data.clone().detach()

                # Initialize test input
                test_input_BALD_BALD = (
                    test_input_perm.clone().detach()
                )  # N_test x obs_dim

                # Record old train data
                observed_train_BALD_old = observed_train_BALD.clone().detach()

                # Active selection BALD
                flag_weight = int((counter_loop + 1) % 2)
                observed_train_BALD, train_pool_data_BALD, flag_full, num_selected = (
                    Active_BALD.base_active_learning_decoder(
                        balance_prop=balance_coef,
                        coef=BALD_coef,
                        observed_train=observed_train_BALD,
                        pool_data=train_pool_data_BALD,
                        step=num_selected_variable,
                        flag_initial=flag_init_train,
                        sigma_out=sigma_out,
                        W_sample=W_sample,
                        strategy="Opposite_Alpha",
                        strategy_alternating=flag_weight,
                        Select_Split=True,
                        selection_scheme=selection_scheme,
                        idx_start=idx_start,
                        idx_end=idx_end,
                    )
                )

                observed_train_BALD = observed_train_BALD.clone().detach()
                train_pool_data_BALD = train_pool_data_BALD.clone().detach()

                # If hybrid, apply the target variable
                if flag_hybrid:
                    observed_train_BALD = Active_BALD.get_target_variable(
                        observed_train_BALD,
                        observed_train_BALD_old,
                        target_dim=-1,
                        train_data=train_data_perm,
                    )
                # Update the selected points
                counter_selection += num_selected

                # Remove the zeros (rz) in the training data
                observed_train_BALD_rz = remove_zero_row_2D(observed_train_BALD)

                # Now Retrain the model
                # redefine the model
                PNP_SGHMC = Point_Net_Plus_BNN_SGHMC(
                    latent_dim=latent_dim,
                    obs_dim=obs_dim,
                    dim_before_agg=dim_before_agg,
                    encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                    encoder_hidden_before_agg=encoder_hidden_before_agg,
                    encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                    encoder_hidden_after_agg=encoder_hidden_after_agg,
                    embedding_dim=embedding_dim,
                    decoder_layer_num=decoder_layer_num,
                    decoder_hidden=decoder_hidden,
                    pooling=pooling,
                    output_const=output_const,
                    add_const=add_const,
                    sample_z=sample_z,
                    sample_W=sample_W,
                    W_sigma_prior=W_sigma_prior,
                    pooling_act=pooling_act,
                    flag_log_q=flag_log_q,
                )

                # Infer Model
                Infer_SGHMC = SGHMC(model=PNP_SGHMC, Infer_name="Scale Adapted SGHMC")
                # Parameter List
                list_p_z = (
                    list(PNP_SGHMC.encoder_before_agg.parameters())
                    + list(PNP_SGHMC.encoder_after_agg.parameters())
                    + [PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias]
                )

                Adam_encoder = torch.optim.Adam(
                    list(PNP_SGHMC.encoder_before_agg.parameters())
                    + list(PNP_SGHMC.encoder_after_agg.parameters()),
                    lr=cfg["Optimizer"]["lr_sghmc"],
                    betas=(cfg["Optimizer"]["beta1"], cfg["Optimizer"]["beta2"]),
                    weight_decay=cfg["Optimizer"]["weight_decay"],
                )
                Adam_embedding = torch.optim.Adam(
                    [PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias],
                    lr=cfg["Optimizer"]["lr_sghmc"],
                    betas=(cfg["Optimizer"]["beta1"], cfg["Optimizer"]["beta2"]),
                    weight_decay=cfg["Optimizer"]["weight_decay"],
                )

                # Define Active obj
                # Random Seed?
                Active_BALD = base_Active_Learning_SGHMC_Decoder(
                    model=PNP_SGHMC,
                    Infer_model=Infer_SGHMC,
                    overall_data=dataset.Data_mat,
                    rs=42,
                    sigma_out=sigma_out,
                    Optim_settings=cfg["Optimizer"],
                    Adam_encoder=Adam_encoder,
                    Adam_embedding=Adam_embedding,
                    flag_clear_target_train=flag_clear_target_train,
                    flag_clear_target_test=flag_clear_target_test,
                    model_name="SGHMC Active",
                )

                # Train the model
                W_sample = Active_BALD.train_BNN(
                    observed_train_BALD_rz,
                    eps=step_sghmc,
                    max_sample_size=40,
                    tot_epoch=epoch_orig + 500,
                    thinning=10,
                    hyper_param_update=25000,
                    sample_int=10,
                    flag_hybrid=flag_hybrid,
                    Adam_encoder=Adam_encoder,
                    Adam_embedding=Adam_embedding,
                    Drop_p=Drop_orig,
                    list_p_z=list_p_z,
                    test_input_tensor=test_pool_data_BALD_BALD.clone().detach(),
                    test_target_tensor=test_target_data.clone().detach(),
                    conditional_coef=conditional_coef_sghmc,
                    target_dim=-1,
                    flag_reset_optim=flag_reset_optim,
                    W_dict_init=W_dict_init,
                    valid_data=valid_data_input_tensor,
                    valid_data_target=valid_data_target_tensor,
                    sigma_out=sigma_out,
                    scale_data=scale_data,
                    noisy_update=noisy_update,
                )

                # Evaluation code
                # RMSE_BALD_BALD, MAE_BALD_BALD, NLL_BALD_BALD = test_UCI_AL(
                #     model=Active_BALD, max_selection=max_selection_eval,sample_x=None,
                #     test_input=test_input_BALD_BALD,test_pool=test_pool_data_BALD_BALD,
                #     test_target=test_target_data, sigma_out=sigma_out,search='Target',
                #     model_name='PNP_SGHMC', W_sample=W_sample)

                # AUIC_BALD_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_BALD_BALD)
                # Results_list = ['%s' % (counter_loop * step), '%.3f' % (AUIC_BALD_BALD)]
                # print(Results_list)

                counter_loop += 1

        # Save results.
        # with open(f'results/{args.dataset}_{args.method}_{trial}.pkl', 'wb') as f:
        #     pickle.dump(results_dict, f)
