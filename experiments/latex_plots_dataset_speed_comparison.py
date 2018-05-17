import numpy as np
from collections import OrderedDict
import copy

general_table_template = \
"""\\documentclass[]{{article}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{multirow}}
\\usepackage{{amssymb}}
\\newcommand{{\\kmeans}}{{$k$-means}}
\\newcommand{{\\B}}{{\\mathbb{{B}}}}
\\usepackage{{mathtools}}

\\begin{{document}}
    \\begin{{centering}}
      \\input{{{py_sub_filename}}}
    \\end{{centering}}
\\end{{document}}
"""

table_template = \
"""
\\newcommand{{\\specialcell}}[2][c]{{%
  \\begin{{tabular}}[#1]{{@{{}}c@{{}}}}#2\\end{{tabular}}}}
  
\\newcolumntype{{H}}{{>{{\\setbox0=\\hbox\\bgroup}}c<{{\\egroup}}@{{}}}}

\\begin{{tabular}}{{{no_columns}}}
\\toprule
\\multicolumn{{{no_info_columns}}}{{c}}{{}} & \\multicolumn{{{no_algo_columns}}}{{c}}{{Speed-up relative to \\kmeans}} \\\\
\\cmidrule{{{start_algo_columns}-{end_algo_columns}}}

{header_cells} \\\\
\\midrule
{dataset_lines}
\\bottomrule
\\end{{tabular}}
"""

"""
data is a list of tuples [('column_name1', list_of_data), ('column_name2': list_of_data), ...]
list_of_data need to be the same type! 
"""

FLOAT_FORMAT = "%.4f"

import os
import re

def get_table_data(dat):
  algorithms = dat.keys()
  algorithms.sort()
  
  durations = []
  stddevs = []

  for i in range(len(algorithms)):
    alg = algorithms[i]
    
    np_durs = np.array(dat[alg]['duration'].values(), dtype=np.float64)
    
    durations.append(np.mean(np_durs))
    stddevs.append(np.std(np_durs))

  return algorithms, durations, stddevs


#def create_dataset_entries(dataset_name, ):

def complete_algorithms(pdata, algs, remove=False):
    datasets = pdata.keys()
    for i in range(len(datasets)):
      dataset = datasets[i]
      no_clusters_list = pdata[dataset]['results'].keys()
      no_clusters_list.sort()
      

        
      
      for j in range(len(no_clusters_list)):
        no_clusters = no_clusters_list[j]
        
        if remove:
          algos_avail = pdata[dataset]['results'][no_clusters]
          alg_keys = algos_avail.keys()
          for alg in alg_keys:
            if alg == 'kmeans':
              continue
            if alg not in algs:
              del algos_avail[alg]
          
        algorithms, durations, stddevs = get_table_data(pdata[dataset]['results'][no_clusters])
        print algorithms, durations, stddevs
        for alg in algorithms:
          if alg == 'kmeans':
            continue
          if alg not in algs:
            algs[alg] = alg
              


def make_columns_same_length(data_list, algs, general_cells):
  column_lengths = OrderedDict()
  for column in general_cells:
    column_lengths[column] = 0
  for column in algs:
    column_lengths[column] = 0
  
  for table_dict in data_list:
    for column in table_dict['data_dict']:
      for string_element in table_dict['data_dict'][column]:
        if len(string_element) > column_lengths[column]:
          column_lengths[column] = len(string_element)
  
  for table_dict in data_list:
    for column in table_dict['data_dict']:
      for i in range(len(table_dict['data_dict'][column])):
        string_element = table_dict['data_dict'][column][i]
        if len(string_element) < column_lengths[column]:
          table_dict['data_dict'][column][i] = string_element.ljust(column_lengths[column])
  
    
def create_table_data(pdata, algs, general_cells):
    data_list = []
    datasets = pdata.keys()
    datasets.sort()
    for i in range(len(datasets)):
      dataset = datasets[i]
      no_clusters_list = pdata[dataset]['results'].keys()
      no_clusters_list.sort()
      dataset_type = 'small'
      data_dict = OrderedDict()
      for k in general_cells:
        data_dict[k] = ["(na)"] * len(no_clusters_list)
      for k in algs:
        data_dict[k] = ["(na)"] * len(no_clusters_list)
      
      print data_dict
      for j in range(len(no_clusters_list)):
        if j == 0:
          data_dict['dataset'][j] = ("\\multirow{%d}{*}{\\specialcell[c]{%s \\\\{\\scriptsize %d / %d / %d}}}"
                                    % (len(no_clusters_list),
                                       dataset,
                                       pdata[dataset]['infos']['input_samples'],
                                       pdata[dataset]['infos']['input_dimension'],
                                       pdata[dataset]['infos']['input_annz']))
        else:
          data_dict['dataset'][j] = ""
        no_clusters = no_clusters_list[j]
        if no_clusters > 1000 and no_clusters <= 5000:
          dataset_type = 'medium'

        if no_clusters >= 10000:
          dataset_type = 'big'
        
        data_dict['num_clusters'][j] = str(no_clusters)       
        algorithms, durations, stddevs = get_table_data(pdata[dataset]['results'][no_clusters])
        best_algo = np.argmax(durations)
        for m in range(len(algorithms)):
          alg = algorithms[m]
          if alg == 'kmeans':
            continue
          dur = ("%.1f" % float(durations[m])) + ("$\\pm$%.1f" % float(stddevs[m]))

          if m == best_algo:
            dur = "\\textbf{%s}" % dur

          try:
            data_dict[alg][j] = dur
          except:
            print alg, data_dict.keys(), data_dict[alg], j
            raise
      
      table_dict = {'data_dict': data_dict,
                   'dataset_type': dataset_type,
                   'input_dimension': pdata[dataset]['infos']['input_dimension'],
                   'input_samples': pdata[dataset]['infos']['input_samples'],
                   'input_annz': pdata[dataset]['infos']['input_annz']}
      data_list.append(table_dict)
      
    make_columns_same_length(data_list, algs, general_cells)
    
    dataset_type_lines = OrderedDict()
    dataset_type_lines['small'] = None
    dataset_type_lines['medium'] = None
    dataset_type_lines['big'] = None
    for table_dict in data_list:
      if dataset_type_lines[table_dict['dataset_type']] is None:
        dataset_type_lines[table_dict['dataset_type']] = ""
      no_lines = len(table_dict['data_dict']['dataset'])
      for i in range(no_lines):
        line_list = []
        for column in table_dict['data_dict']:
          line_list.append(table_dict['data_dict'][column][i])
        dataset_type_lines[table_dict['dataset_type']] += " & ".join(line_list) + " \\\\\n"
    
    types_to_delete = []
    for dataset_type in dataset_type_lines:
      if dataset_type_lines[dataset_type] is None:
        types_to_delete.append(dataset_type)
    
    for dataset_type in types_to_delete:
      del dataset_type_lines[dataset_type]
    
    return dataset_type_lines
    
def create_table(output_folder=None,
                table_name=None,
                pdata=None,
                algs=OrderedDict(),
                title="tbl-speed-comparison"):
    
    pdata = copy.deepcopy(pdata)
    pname = title
    py_general_filename_tex = pname + "-single.tex"
    py_sub_filename = pname
    py_sub_filename_tex = pname + ".tex"
    
    general_cells = OrderedDict()
    general_cells['dataset'] = "Dataset \\\\  num / dim / annz"
    general_cells['num_clusters'] = "k"
    
    general_table = general_table_template.format(py_sub_filename=py_sub_filename)   
    complete_algorithms(pdata, algs, remove=True)
    
    header_cells = OrderedDict()
    for k in general_cells.keys():
      header_cells[k] = general_cells[k]
      
    for k in algs.keys():
      header_cells[k] = algs[k]

    header_cell_list = []
    for c in header_cells:
      header_cell_list.append("\\specialcell[c]{%s}" % header_cells[c])
    dataset_type_lines = create_table_data(pdata, algs, general_cells)
    header_cells_str = " & ".join(header_cell_list)
    
    no_columns = len(general_cells) * "c" + "r" * len(algs)
      
    dataset_type_lines_keys = dataset_type_lines.keys()
    for i in range(len(dataset_type_lines_keys)):
      dataset_type = dataset_type_lines_keys[i]
      if i != len(dataset_type_lines_keys) - 1:
        dataset_type_lines[dataset_type] += "\n\\midrule\n"
    
    tbl = table_template.format(no_info_columns=len(general_cells),
                                no_algo_columns=len(algs),
                                start_algo_columns=len(general_cells) + 1,
                                end_algo_columns=len(general_cells) + len(algs),
                                header_cells=header_cells_str,
                                no_columns=no_columns,
                                dataset_lines="".join(dataset_type_lines.values()))
    
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
    
    with open(os.path.join(output_folder, py_general_filename_tex), 'wb') as f:
      f.write(general_table)
      
    with open(os.path.join(output_folder, py_sub_filename_tex), 'wb') as f:
      f.write(tbl)
      
    # create list of lists which contain the contents to write
     