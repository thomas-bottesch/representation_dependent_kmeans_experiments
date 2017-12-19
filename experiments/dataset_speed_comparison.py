from __future__ import print_function
from latex_plots_dataset_speed_comparison import create_plot
from experiment_db import ExperimentDB
from pprint import pprint

def remove_where_kmeans_not_complete(result_data):
  remove_cluster_data = False
  
  if 'kmeans' not in result_data:
    remove_cluster_data = True
  else:
      reference_run_set = set([0,1,2])
      if set(result_data['kmeans']['duration'].keys()) != reference_run_set:
        remove_cluster_data = True
      else:
        algs_to_delete = []
        reference_no_iterations = None
        for alg in result_data:
          if alg == 'kmeans':
            continue
          
          # verify that all runs were made
          if set(result_data[alg]['duration'].keys()) != reference_run_set:
            algs_to_delete.append(alg)
            continue
            
          # verify that all algorithms had exactly the same number of iterations
          if reference_no_iterations is None:
            reference_no_iterations = result_data[alg]['no_iterations']
          
          if reference_no_iterations != result_data[alg]['no_iterations']:
            pprint(result_data)
            raise Exception("wrong iteration count")
        
        for alg in algs_to_delete:
          del result_data[alg]
        
        if reference_no_iterations is None:
          # only 'kmeans' left
          remove_cluster_data = True
        else:
          no_iterations_calculated_kmeans = 3.0
          
          for run in reference_no_iterations:
            kmeans_duration = result_data['kmeans']['duration'][run]
            kmeans_duration_per_iter = kmeans_duration / no_iterations_calculated_kmeans
            result_data['kmeans']['duration'][run] = kmeans_duration_per_iter * reference_no_iterations[run]
            
            for alg in result_data:
              if alg == 'kmeans':
                continue
              
              result_data[alg]['duration'][run] = result_data['kmeans']['duration'][run] / result_data[alg]['duration'][run] 
            
            result_data['kmeans']['duration'][run] = 1.0
          
        
  return remove_cluster_data

def remove_incomplete_data(result_data):
  
  datasets_to_delete = []
  
  for ds in result_data:
    clusters_to_delete = []
    for no_clusters in result_data[ds]['results']:
      print(ds, no_clusters)
      if remove_where_kmeans_not_complete(result_data[ds]['results'][no_clusters]):
        clusters_to_delete.append(no_clusters)
    
    for no_clusters in clusters_to_delete:
      del result_data[ds]['results'][no_clusters]
    
    if len(result_data[ds]['results']) == 0:
      datasets_to_delete.append(ds)
  
  for ds in datasets_to_delete:
    del result_data[ds]

def result_evaluation_dataset_speed_comparison(out_folder, out_folder_csv):
  
  for fcnt, plotname in [('do_kmeans', 'kmeans_speeds')]:
    print(plotname)
    run_identifiers = ExperimentDB.get_identifiers(out_folder, fcnt)
    plot_data = {}
    
    result_data = {}
    for run_identifier in run_identifiers:
      db = ExperimentDB(out_folder, fcnt, run_identifier)
      
      for resid in db.get_algorithm_run_ids():
        (control_params, params, res) = db.get_experiment_result_from_run_id(resid)
        if res is None:
          continue
        ds = params['info']['dataset_name']
        alg = params['info']['algorithm']
        no_clusters = params['task']['no_clusters']
        run = params['task']['run']
        duration_kmeans = res['duration_kmeans']
        no_iterations = len(res['iteration_changes'])
        
        if ds not in result_data:
          result_data[ds] = {}
          result_data[ds]['results'] = {}
          result_data[ds]['infos'] = {}
          
        if no_clusters not in result_data[ds]['results']:
          result_data[ds]['results'][no_clusters] = {}
          
        if alg not in result_data[ds]['results'][no_clusters]:
          result_data[ds]['results'][no_clusters][alg] = {}
          
        if 'duration' not in result_data[ds]['results'][no_clusters][alg]:
          result_data[ds]['results'][no_clusters][alg]['duration'] = {}
          
        if 'no_iterations' not in result_data[ds]['results'][no_clusters][alg]:
          result_data[ds]['results'][no_clusters][alg]['no_iterations'] = {}
        
        result_data[ds]['results'][no_clusters][alg]['duration'][run] = duration_kmeans
        
        if 'truncated_svd' in res:
          result_data[ds]['results'][no_clusters][alg]['duration'][run] += res['truncated_svd']['duration']
        
        result_data[ds]['infos']['input_dimension'] = res['input_dimension']
        result_data[ds]['infos']['input_samples'] = res['input_samples']
        result_data[ds]['infos']['input_annz'] = res['input_annz']
        result_data[ds]['results'][no_clusters][alg]['no_iterations'][run] = no_iterations
      
      remove_incomplete_data(result_data)
      
      print("Result data:")
      pprint(result_data)
    
    create_plot(output_folder=out_folder_csv,
                plot_name=plotname,
                pdata=result_data)