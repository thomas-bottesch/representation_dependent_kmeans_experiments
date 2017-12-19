from __future__ import print_function
import argparse
import os
import collections
import traceback
import time
from experiment_db import ExperimentDB
from sklearn.decomposition import TruncatedSVD
from Queue import Empty
from multiprocessing import Process, Queue
from os.path import abspath, join, dirname
from collections import OrderedDict
from fcl import kmeans
from fcl.datasets import load_sector_dataset, load_usps_dataset, load_and_extract_dataset_from_github
from fcl.matrix.csr_matrix import get_csr_matrix_from_object, csr_matrix_to_libsvm_string
from dataset_speed_comparison import result_evaluation_dataset_speed_comparison
from memory_consumption import result_evaluation_memory_consumption


# params contains parameter which modify the internals of the funtion_to_evaluate
# if params get changed and do not exist in the database, funtion_to_evaluate is
# run for the params
# control_params contain data which is needed for function to evaluate 
# e.g. control_params contains {dataset_path: <..>, dataset_name: <..>, out_folder: <..>]}
# params contain:
# params_general['tsne'] = collections.OrderedDict()
# params_general['tsne']['tsne_components'] = "2"
# params_general['tsne']['perplexity'] = "30."
# params_general['tsne']['theta'] = "0.5"
# params_general['tsne']['algorithm'] = "bh_sne"
# a change in control params does not lead to a reevaluation
# run_identifier needed to determine the current database. often the dataset name is used 
def evaluate(run_identifier, control_params, params, function_to_evaluate, out_folder, force = False):
  try:
    experiment_db = ExperimentDB(out_folder, function_to_evaluate.__name__, run_identifier, dump_also_as_json=True)
    previous_result = experiment_db.get_experiment_result(params)
    
    if previous_result is not None and not force:
      print("already_exists:", function_to_evaluate.__name__ + " for", run_identifier, "with", params)
      return previous_result
    else:
      print(function_to_evaluate.__name__ + " for", run_identifier, "with", params)
      result_q = Queue()
      p = Process(target=function_to_evaluate, args=(result_q, control_params, params))
      p.start()
      p.join() # this blocks until the process terminates
      
      try:
        res = result_q.get_nowait()
      except Empty:
        print("no result available for this call. the process most likely failed with %s" % str(p.exitcode))
        res = None
      
      experiment_db.add_experiment(control_params, params, res)
      return (control_params, params, res)
  except:
    raise


def do_kmeans(result_q, control_params, params):
    
    X = control_params['libsvm_dataset_path']
    
    data_as_csrmatrix = get_csr_matrix_from_object(X)
    no_samples, _ = data_as_csrmatrix.shape
    print(no_samples)
    
    info = params['info']
    task = params['task']
    
    output = "for %s with algorithm=%s run=%d k=%d"%(info['dataset_name'],
                                                              info['algorithm'],
                                                              task['run'],
                                                              task['no_clusters'])
    
    if 'pca' in info['algorithm']:
      
      annz_input_matrix = data_as_csrmatrix.annz
      desired_no_eigenvectors = int(data_as_csrmatrix.annz * info['truncated_svd_annz_percentage'])
      print("Using TruncatedSVD to retrieve %d eigenvectors from input matrix with %d annz" % (desired_no_eigenvectors,
                                                                                               annz_input_matrix))
      p = TruncatedSVD(n_components = int(data_as_csrmatrix.annz * info['truncated_svd_annz_percentage']))
      start = time.time()
      scipy_csr_matrix = data_as_csrmatrix.to_numpy()
      p.fit(scipy_csr_matrix)
      # convert to millis
      fin = (time.time() - start) * 1000 
      pca_projection_csrmatrix = get_csr_matrix_from_object(p.components_)
      (no_components, no_features) = p.components_.shape
      print("Time needed to complete getting %d eigenvectors with %d features with SVD:" % (no_components, no_features),
            fin, "(annz of the top eigenvectors:", pca_projection_csrmatrix.annz, ")")
      additional_algo_data = {info['algorithm']: {'data': pca_projection_csrmatrix, 'duration': fin}}
    else:
      additional_algo_data = {}
    
    print("Executing " + output)
    km = kmeans.KMeans(n_jobs=1, no_clusters=task['no_clusters'], algorithm=info['algorithm'],
                       init='random', seed = task['run'], verbose = True, additional_params = dict(task),
                       iteration_limit = task['iteration_limit'], additional_info = dict(info))
    
    if info['algorithm'] in additional_algo_data:
      km.fit(data_as_csrmatrix, external_vectors = additional_algo_data[info['algorithm']]['data'])
      result = km.get_tracked_params()
      result['truncated_svd'] = {}
      result['truncated_svd']['no_components'] = no_components
      result['truncated_svd']['no_features'] = no_features
      result['truncated_svd']['duration'] = additional_algo_data[info['algorithm']]['duration']
    else:
      km.fit(data_as_csrmatrix)
      result = km.get_tracked_params()
    
    result_q.put(result)
        
def do_evaluations(dataset_path, dataset_name, out_folder, params, clusters):
  
  algorithms = ["kmeans_optimized", "pca_kmeans", "elkan", "kmeans", "yinyang", "pca_elkan", 'fast_yinyang']
  
  control_params = OrderedDict()
  control_params['libsvm_dataset_path'] = dataset_path

  if 'calculate_kmeans' in params:
    for no_clusters in clusters:
      for algorithm in algorithms:
        
        for r in range(3):
          params['calculate_kmeans']['task'] = collections.OrderedDict()
          params['calculate_kmeans']['task']['no_clusters'] = no_clusters
          params['calculate_kmeans']['task']['run'] = r
          if algorithm == 'kmeans':
            # for kmeans we only do 3 iterations
            # we then calculate the average duration of these three iterations 
            # and extrapolate to retrieve the time needed to calculate
            # the full kmeans
            params['calculate_kmeans']['task']['iteration_limit'] = 3
          else:
            params['calculate_kmeans']['task']['iteration_limit'] = 1000
                   
          params['calculate_kmeans']['info'] = collections.OrderedDict()
          params['calculate_kmeans']['info']['dataset_name'] = dataset_name
          params['calculate_kmeans']['info']['algorithm'] = algorithm
          params['calculate_kmeans']['info']['truncated_svd_annz_percentage'] = 0.1
          
          try:
            evaluate("kmeans", control_params, params['calculate_kmeans'], do_kmeans, out_folder)
          except:
            print(traceback.format_exc()) 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment to evaluate k-means variants with feature maps that exploit'
                                               ' different representations of data sets')
  parser.add_argument('--dataset_folder', type=str, default="datasets", help='Path datasets are downloaded to')
  parser.add_argument('--output_path', type=str, default="output_path", help='Path to the results of single algorithm executions')
  parser.add_argument('--output_path_latex', type=str, default="output_path_latex", help='Path to the results as latex tables')
  parser.add_argument('--only_result_evaluation', dest='only_evaluation', action='store_true',
                      help='Recreate latex tables based on previous results without executing kmeans again')
  parser.add_argument('--testmode', dest='testmode', action='store_true', help='Only run the experiments for a single small dataset')
  parser.set_defaults(only_evaluation=False)
  parser.set_defaults(testmode=False)
  
  args = parser.parse_args()
  
  out_folder = abspath(args.output_path)
  output_path_latex = abspath(args.output_path_latex)
  
  if not args.only_evaluation:
    ds_folder = abspath(args.dataset_folder)
    
    if not os.path.isdir(ds_folder):
      os.makedirs(ds_folder)
    
    if not os.path.isdir(out_folder):
      os.makedirs(out_folder)
       
    params_general = collections.OrderedDict()
    params_general['calculate_kmeans'] = collections.OrderedDict()
    clusters_big = [100, 1000, 10000]
    clusters_medium = [100, 500, 5000]
    clusters_small = [100, 250, 1000]
    
    do_evaluations(load_usps_dataset(ds_folder), 'usps', out_folder, params_general, clusters_small)
    if not args.testmode:
      do_evaluations(load_sector_dataset(ds_folder), 'sector', out_folder, params_general, clusters_small)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'real_sim.scaled.bz2'), 'realsim', out_folder, params_general, clusters_medium)    
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'mediamill_static_label_scaled.bz2'), 'mediamill', out_folder, params_general, clusters_medium)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'caltech101.scaled.bz2'), 'caltech101', out_folder, params_general, clusters_big)  
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'e2006_static_label.scaled.bz2'), 'e2006', out_folder, params_general, clusters_small)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'avira_201.scaled.bz2'), 'avira201', out_folder, params_general, clusters_big)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'kdd.scaled.bz2'), 'kdd2001', out_folder, params_general, clusters_big)
      do_evaluations(load_and_extract_dataset_from_github('fcl_datasets2', ds_folder, 'mnist800k.scaled.bz2'), 'mnist800k', out_folder, params_general, clusters_big)
  else:
    if not os.path.isdir(out_folder):
      raise Exception("cannot do evaluation with nonexisting output dir %s" % out_folder)

  if not os.path.isdir(output_path_latex):
    os.makedirs(output_path_latex)

  result_evaluation_dataset_speed_comparison(out_folder, output_path_latex)
  result_evaluation_memory_consumption(out_folder, output_path_latex)
