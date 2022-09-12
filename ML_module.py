#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220830

import json

import feature_explain as fe
import predict as ped
import train as t


def run(ctrl, p):
    # get param
    with open(p) as f:
        param_file = json.load(f)

    # train
    if int(ctrl[0]):
        best_param = t.run(param_file['train_params'])
        print('Train Done')
    else:
        best_param = ''

    # feature explain
    if int(ctrl[1]):
        # update params
        ex_params = param_file['explain_params']
        if best_param != '':
            ex_params.update(best_param)
        print(ex_params)
        fe.run(ex_params)
        print('Feature Explain Done')

    # predict
    if int(ctrl[2]):
        # update params
        pred_params = param_file['predict_params']
        print(pred_params)
        ped.run(pred_params)
        print('Predict Done')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument(
        '-f', '--file_path',
        dest='FilePath',
        help='json file path, ex:"example.json"')

    parser.add_argument(
        '-r', '--run_project',
        dest='RUNMODUEL',
        help='string code, ex:111')

    args = parser.parse_args()

    run(ctrl=args.RUNMODUEL,
        p=args.FilePath)


if __name__ == '__main__':
    main()
    '''
    run(ctrl='01',
        p='C:/Users/wang/Desktop/'
          'ML_Paper/params/params_sample.json')
    '''
