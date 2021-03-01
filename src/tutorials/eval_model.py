"""
A sample to evaluate model on dataset
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import sys
sys.path.append("../")

import argparse
import numpy as np
import os

from models.mnist_utils import load_pool, load_lenet
from utils.file import load_from_json
from models.utils.estimator import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def evaluate_cnn(trans_configs, model_configs,
                 data_configs, save=False, output_dir=None):
    """
    Apply transformation(s) on images.
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # get the CNN undefended model (UM)
    cnn_configs = model_configs.get('cnn')
    file = os.path.join(cnn_configs.get('dir'), cnn_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load weak defenses into a pool
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=cnn_configs,
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble from the WD pool
    wds = list(pool.values())
    print(">>> wds:", type(wds), type(wds[0]))
    ensemble = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)
    img_rows, img_cols = x_bs.shape[1], x_bs.shape[2]

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)

    # Evaluate AEs.
    results = {}
    ae_list = data_configs.get('ae_files')
    ae_file = os.path.join(data_configs.get('dir'), ae_list[4])
    x_adv = np.load(ae_file)

    # evaluate the undefended model on the AE
    print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
    pred_adv_um = undefended.predict(x_adv)
    err_um = error_rate(y_pred=pred_adv_um, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['UM'] = err_um

    # evaluate the ensemble on the AE
    print(">>> Evaluating ensemble on [{}], it may take a while...".format(ae_file))
    pred_adv_ens = ensemble.predict(x_adv)
    err_ens = error_rate(y_pred=pred_adv_ens, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['Ensemble'] = err_ens

    # evaluate the baseline on the AE
    print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
    pred_adv_bl = baseline.predict(x_adv)
    err_bl = error_rate(y_pred=pred_adv_bl, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['PGD-ADT'] = err_bl

    # TODO: collect and dump the evaluation results to file(s) such that you can analyze them later.
    print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))


def evaluate_svm(trans_configs, model_configs,
                 data_configs, save=False, output_dir=None):
    """
    Apply transformation(s) on images.
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # get the CNN undefended model (UM)
    # Since the uploaded AEs were generated for the CNN UM,
    # collect the corrections based on the CNN UM.
    cnn_configs = model_configs.get('cnn')
    file = os.path.join(cnn_configs.get('dir'), cnn_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load weak defenses into a pool
    svm_configs = model_configs.get('svm')
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=svm_configs,
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble from the WD pool
    wds = list(pool.values())
    print(">>> wds:", type(wds), type(wds[0]))
    ensemble = Ensemble(classifiers=wds,
                        strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)
    img_rows, img_cols = x_bs.shape[1], x_bs.shape[2]

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)

    # Evaluate AEs.
    results = {}
    ae_list = data_configs.get('ae_files')
    ae_file = os.path.join(data_configs.get('dir'), ae_list[4])
    x_adv = np.load(ae_file)

    # evaluate the undefended model on the AE
    print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
    pred_adv_um = undefended.predict(x_adv)
    err_um = error_rate(y_pred=pred_adv_um, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['UM'] = err_um

    # evaluate the ensemble on the AE
    print(">>> Evaluating ensemble on [{}], it may take a while...".format(ae_file))
    pred_adv_ens = ensemble.predict(x_adv)
    err_ens = error_rate(y_pred=pred_adv_ens, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['Ensemble'] = err_ens

    # evaluate the baseline on the AE
    print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
    pred_adv_bl = baseline.predict(x_adv)
    err_bl = error_rate(y_pred=pred_adv_bl, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['PGD-ADT'] = err_bl

    # TODO: collect and dump the evaluation results to file(s) such that you can analyze them later.
    print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))


def evaluate_hybrid(trans_configs, model_configs,
                    data_configs, save=False, output_dir=None):
    """
    Apply transformation(s) on images.
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # get the CNN undefended model (UM)
    # Since the uploaded AEs were generated for the CNN UM,
    # collect the corrections based on the CNN UM.
    cnn_configs = model_configs.get('cnn')
    file = os.path.join(cnn_configs.get('dir'), cnn_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load CNN weak defenses into a pool
    cnn_pool, _ = load_pool(trans_configs=trans_configs,
                            model_configs=cnn_configs,
                            active_list=True,
                            wrap=True)

    cnns = list(cnn_pool.values())
    print(">>> CNNs:", type(cnns), type(cnns[0]))

    # load SVM weak defenses into a pool
    svm_configs = model_configs.get('svm')
    svm_pool, _ = load_pool(trans_configs=trans_configs,
                            model_configs=svm_configs,
                            active_list=True,
                            wrap=True)

    svms = list(svm_pool.values())
    print(">>> SVMs:", type(svms), type(svms[0]))

    # ----------------------
    # Hybrid
    # ----------------------
    # There are multiple ways to select weak defenses for your ensemble.
    # I shew you a naive strategy here.
    # Try to come up your own strategy, so you will create several variants of
    # hybrid ensembles. Evaluate all your ensemble variants.
    # A naive strategy to build hybrid variants:
    # 1. fix the number of total weak defenses in the hybrid ensemble.
    # 2. build a hybrid ensemble, starting from all CNN weak defenses.
    # 3. build your next hybrid ensemble by replacing part of the CNN weak defenses with SVM weak defenses.
    #    For examples, replace 10%, then 20%, 30%, etc. of the weak defenses with SVMs,
    #    until you get a hybrid consisting only SVMs.
    # 4. So, you will have 11 ensemble variants, varying the ratio between two types of models.
    """
    select weak defenses for your ensemble
    in this example, we build a hybrid ensemble consisting of
    3 cnn models + 3 svm models
    """
    wds = cnns
    wds.extend(svms)
    ensemble = Ensemble(classifiers=wds,
                        strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # -----------------------------
    # Evaluate models
    # -----------------------------
    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)
    img_rows, img_cols = x_bs.shape[1], x_bs.shape[2]

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)

    # Evaluate AEs.
    results = {}
    ae_list = data_configs.get('ae_files')
    ae_file = os.path.join(data_configs.get('dir'), ae_list[4])
    x_adv = np.load(ae_file)

    # evaluate the undefended model on the AE
    print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
    pred_adv_um = undefended.predict(x_adv)
    err_um = error_rate(y_pred=pred_adv_um, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['UM'] = err_um

    # evaluate the ensemble on the AE
    print(">>> Evaluating ensemble on [{}], it may take a while...".format(ae_file))
    pred_adv_ens = ensemble.predict(x_adv)
    err_ens = error_rate(y_pred=pred_adv_ens, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['Ensemble'] = err_ens

    # evaluate the baseline on the AE
    print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
    pred_adv_bl = baseline.predict(x_adv)
    err_bl = error_rate(y_pred=pred_adv_bl, y_true=labels, correct_on_bs=corrections)
    # track the result
    results['PGD-ADT'] = err_bl

    # TODO: collect and dump the evaluation results to file(s) such that you can analyze them later.
    print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    """
    configurations under ``configs/demo`` are for demo.
    """

    parser.add_argument('-t', '--trans-configs', required=False,
                        default='../configs/demo/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-m', '--model-configs', required=False,
                        default='../configs/demo/hybrid-mnist.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='../configs/demo/data-mnist.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()

    print('------AUGMENT SUMMARY-------')
    print('TRANSFORMATION CONFIGS:', args.trans_configs)
    print('MODEL CONFIGS:', args.model_configs)
    print('DATA CONFIGS:', args.data_configs)
    print('OUTPUT ROOT:', args.output_root)
    print('DEBUGGING MODE:', args.debug)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    trans_configs = load_from_json(args.trans_configs)
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)

    # -------- Evaluate CNN ATHENA -------------
    # evaluate_cnn(trans_configs=trans_configs,
    #              model_configs=model_configs,
    #              data_configs=data_configs,
    #              save=False,
    #              output_dir=args.output_root)

    # -------- Evaluate SVM ATHENA -------------
    # evaluate_svm(trans_configs=trans_configs,
    #              model_configs=model_configs,
    #              data_configs=data_configs,
    #              save=False,
    #              output_dir=args.output_root)

    # -------- Evaluate HYBRID ATHENA -------------
    evaluate_hybrid(trans_configs=trans_configs,
                    model_configs=model_configs,
                    data_configs=data_configs,
                    save=False,
                    output_dir=args.output_root)