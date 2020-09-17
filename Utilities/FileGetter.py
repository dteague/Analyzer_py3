#!/usr/bin/env python3
import os
import json
import imp
import glob
import argparse

class FileGetter:
    def __init__(self, analysis, preselection):
        try:
            adm_path = os.environ['ADM_PATH']
        except:
            print('The Analysis Dataset Manager is found by the variable ADM_PATH')
            print('Please set this path and consider setting it in your .bashrc')
            exit(1)
        #adm_path = '

        self.analysis = analysis
        self.preselection = preselection
        self.groupInfo = self.readAllInfo("{}/PlotGroups/{}.py"
                                          .format(adm_path, analysis))
        self.mcInfo = self.readAllInfo(
            "{}/FileInfo/montecarlo/montecarlo_2016.py".format(adm_path))
        self.fileInfo = self.readAllInfo(
            "{}/FileInfo/{}/{}.py".format(adm_path, analysis, preselection))
        self.group2MemeberMap = {key: item["Members"]
                                 for key, item in self.groupInfo.items()}


    def readAllInfo(self, file_path):
        info = {}
        for info_file in glob.glob(file_path):
            file_info = self.readInfo(info_file)
            if file_info:
                info.update(file_info)
        return info

    def readInfo(self, file_path):
        if ".py" not in file_path[-3:] and ".json" not in file_path[-5:]:
            if os.path.isfile(file_path + ".py"):
                file_path = file_path + ".py"
            elif os.path.isfile(file_path + ".json"):
                file_path = file_path + ".json"
            else:
                return
        if ".py" in file_path[-3:]:
            file_info = imp.load_source("info_file", file_path)
            info = file_info.info
        else:
            info = self.readJson(file_path)
        return info

    def readJson(self, json_file_name):
        json_info = {}
        with open(json_file_name) as json_file:
            try:
                json_info = json.load(json_file)
            except ValueError as err:
                print("Error reading JSON file {}. The error message was:"
                      .format(json_file_name))
                print(err)
        return json_info

    def get_file_dict(self, group_list=None):
        if group_list is None:
            return {key: item["file_path"] for key, item in self.fileInfo.items()}

        else:
            return_dict = dict()
            for group in group_list:
                if group in self.group2MemeberMap:
                    return_dict.update({sample: self.fileInfo[sample]["file_path"]
                                        for sample in self.group2MemeberMap[group]})
                else:
                    return_dict[group] = self.fileInfo[group]["file_path"]
            return return_dict

    def get_xsec(self, group):
        scale = self.mcInfo[group]['cross_section']
        if "kfactor" in self.mcInfo[group]:
            scale *= self.mcInfo[group]["kfactor"]
        return scale


def get_generic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("proc_type", type=str,)
    parser.add_argument("-o", "--outdir", type=str, required=True,
                        help="directory where to write masks")
    parser.add_argument("-a", "--analysis", type=str, required=True,
                        help="Specificy analysis used")
    parser.add_argument("-s", "--selection", type=str, required=True,
                        help="Specificy selection used")
    parser.add_argument("-c", "--channel", type=str, default="",
                        help="Channels to run over")
    parser.add_argument("-j", type=int, default=1, help="Number of cores")
    parser.add_argument("-r", action="store_true", help="Remake create files")
    parser.add_argument("-f", "--filenames", required=True,
                        type=lambda x : [i.strip() for i in x.split(',')],
                        help="List of input file names, "
                        "as defined in AnalysisDatasetManager, separated "
                        "by commas")
    return parser.parse_args()

def checkOrCreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pre(pre, lister):
    return ["_".join([pre, l]) for l in lister]
