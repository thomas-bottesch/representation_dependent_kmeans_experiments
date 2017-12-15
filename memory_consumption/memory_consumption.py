from __future__ import print_function
import argparse
import os
import json
import collections
import numpy
import inspect
import cPickle
import traceback
import numpy as np
import time
from copy import deepcopy
from pprint import pprint
from sklearn.decomposition import TruncatedSVD
from Queue import Empty
from multiprocessing import Process, Queue
from os.path import abspath, join, dirname
from collections import OrderedDict
from latex_plots import create_plot
from fcl import kmeans
from fcl.datasets import load_sector_dataset, load_usps_dataset, load_and_extract_dataset_from_github
from fcl.matrix.csr_matrix import get_csr_matrix_from_object, csr_matrix_to_libsvm_string


class ExperimentDB():
  
  def __init__(self, out_folder, function_name, run_identifier, dump_also_as_json=False):
    self.output_folder_local = join(out_folder, function_name)
    self.run_identifier = run_identifier
    self.dump_also_as_json = dump_also_as_json
    if not os.path.isdir( self.output_folder_local):
      os.makedirs(self.output_folder_local)
    
    self.output_file_path_database = join(self.output_folder_local, "index" + "_" + self.run_identifier)
    if os.path.isfile(self.output_file_path_database):
      with open(self.output_file_path_database, 'rb') as f:
        self.db = json.load(f)
        self.next_algorithm_run_id = self.db[-1][0] + 1
    else:
      self.db = []
      self.next_algorithm_run_id = 0
  
  def get_ikeys(self):
    return [int(k) for k in self.db.keys()]
  
  def get_algorithm_run_ids(self):
    return [self.db[k][0] for k in range(len(self.db))]
  
  @classmethod
  def get_identifiers(cls, out_folder, function_name):
    run_identifiers = []
    output_folder_local = join(out_folder, function_name)
    for f in os.listdir(output_folder_local):
      if "index_" in f:
        run_identifier = f.split("_")[1]
        run_identifiers.append(run_identifier)
    return run_identifiers
  
  def get_last_experiment_result_only(self):
    if len(self.db) == 0:
      return None
    else:
      (control_params, params, res) = self.get_experiment_result_from_run_id(self.db[-1][0])
      return res
  
  # retrieves the experiment result for a set of parameters if it exists
  # returns None if for this set of parameters no experiment was done
  def get_experiment_result(self, params):
    previous_runs = [algorithm_run_id for algorithm_run_id, prms in self.db if params == prms]
    if len(previous_runs) != 0:
      algorithm_run_id = previous_runs[0]
    else:
      return None
    
    output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(algorithm_run_id)) 
    with open(output_file_run, 'rb') as f:
      return cPickle.load(f)
    
  # retrieves the experiment result for a set of parameters if it exists
  # returns None if for this set of parameters no experiment was done
  def get_experiment_result_from_run_id(self, algorithm_run_id):
    output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(algorithm_run_id)) 
    with open(output_file_run, 'rb') as f:
      return cPickle.load(f)
  
  def persist_db(self):
    with open(self.output_file_path_database, 'wb') as f:
      json.dump(self.db, f, indent=4)
  
  def add_experiment(self, control_params, params, res):
    self.db.append((self.next_algorithm_run_id,params))
    
    output_file_run = join(self.output_folder_local, self.run_identifier + "_" + str(self.next_algorithm_run_id)) 
    with open(output_file_run, 'wb') as f:
      cPickle.dump((control_params, params, res), f)
    
    if self.dump_also_as_json:
      output_file_run_json = output_file_run + ".txt"
      with open(output_file_run_json, 'wb') as f:
        json.dump(res, f, indent=4)
      
    self.next_algorithm_run_id += 1
    self.persist_db()


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
            kmeans_memory_consumption = result_data['kmeans']['duration'][run][0]
            kmeans_duration = result_data['kmeans']['duration'][run][1]
            kmeans_duration_per_iter = kmeans_duration / no_iterations_calculated_kmeans
            kmeans_dur =  kmeans_duration_per_iter * reference_no_iterations[run]
            
            for alg in result_data:
              if alg == 'kmeans':
                continue
              alg_consumption = result_data[alg]['duration'][run][0]
              alg_dur = result_data[alg]['duration'][run][1]
              result_data[alg]['duration'][run] = (alg_consumption, kmeans_dur / alg_dur) 
            
            result_data['kmeans']['duration'][run] = (kmeans_memory_consumption, 1.0)
        
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

def result_evaluation(out_folder, out_folder_csv):
  
  for fcnt, plotname in [(do_kmeans, 'kmeans_speeds')]:
  
  #for fcnt, plotname in [(do_calculate_length_normalized, 'datavisnorm')]:
    print(plotname)
    run_identifiers = ExperimentDB.get_identifiers(out_folder, fcnt.__name__)
    plot_data = {}
    
    result_data = {}
    for run_identifier in run_identifiers:
      db = ExperimentDB(out_folder, fcnt.__name__, run_identifier)
      
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
        
        no_samples = res['input_samples']
        size_of_data_storage_element = 8
        size_of_key_storage_element = 4
        size_of_pointer_storage_element = 8
        
        if alg != 'kmeans':
          no_clusters_remaining = res['no_clusters_remaining']
        
        if alg == 'kmeans':
          mem_consumption = 0
        elif alg == 'elkan':
          # elkan stores two dense matrices
          # 1. lower_bound_matrix = no_samples * no_clusters_remaining
          # 2. distance_between_clusters_matrix = no_clusters_remaining * no_clusters_remaining
          
          lower_bound_matrix_mem_consumption = no_samples * no_clusters_remaining * size_of_data_storage_element
          distance_between_clusters_matrix_mem_consumption = no_clusters_remaining * no_clusters_remaining * size_of_data_storage_element
          mem_consumption = lower_bound_matrix_mem_consumption + distance_between_clusters_matrix_mem_consumption
        elif alg == 'pca_elkan':  
          # pca_elkan stores two dense matrices + orthonormal_basis_matrix + projected_matrix_samples + projected_matrix_clusters
          # 1. lower_bound_matrix = no_samples * no_clusters_remaining
          # 2. distance_between_clusters_matrix = no_clusters_remaining * no_clusters_remaining
          # 3. orthonormal_basis_matrix = no_orthonormal_vectors * orthonormal_basis_matrix_dim
          # 4. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
          # 5  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
          lower_bound_matrix_mem_consumption = no_samples * no_clusters_remaining * size_of_data_storage_element
          distance_between_clusters_matrix_mem_consumption = no_clusters_remaining * no_clusters_remaining * size_of_data_storage_element
          
          # These matrices are stored as sparse matrices. Can be changed in the future since these matrices are almost completely dense 
          orthonormal_basis_matrix_mem_consumption = (res['truncated_svd']['no_components']
                                                      * res['truncated_svd']['no_features']
                                                      * (size_of_data_storage_element + size_of_key_storage_element)) \
                                                      + ((res['truncated_svd']['no_components'] + 1) * size_of_pointer_storage_element)
          projected_matrix_samples_mem_consumption = (no_samples * res['truncated_svd']['no_components']
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
                              
          projected_matrix_clusters_mem_consumption = (no_clusters_remaining * res['truncated_svd']['no_components']
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
          
          mem_consumption = lower_bound_matrix_mem_consumption \
                                                     + distance_between_clusters_matrix_mem_consumption \
                                                     + orthonormal_basis_matrix_mem_consumption \
                                                     + projected_matrix_samples_mem_consumption \
                                                     + projected_matrix_clusters_mem_consumption
        elif alg == 'pca_kmeans':
          # pca_elkan stores a orthonormal_basis_matrix + projected_matrix
          # 1. orthonormal_basis_matrix = no_orthonormal_vectors * orthonormal_basis_matrix_dim
          # 2. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
          # 3  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
          
          # These matrices are stored as sparse matrices. Can be changed in the future since these matrices are almost completely dense 
          orthonormal_basis_matrix_mem_consumption = (res['truncated_svd']['no_components']
                                                      * res['truncated_svd']['no_features']
                                                      * (size_of_data_storage_element + size_of_key_storage_element)) \
                                                      + ((res['truncated_svd']['no_components'] + 1) * size_of_pointer_storage_element)
          projected_matrix_samples_mem_consumption = (no_samples * res['truncated_svd']['no_components']
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
                              
          projected_matrix_clusters_mem_consumption = (no_clusters_remaining * res['truncated_svd']['no_components']
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
          
          mem_consumption = orthonormal_basis_matrix_mem_consumption \
                            + projected_matrix_samples_mem_consumption \
                            + projected_matrix_clusters_mem_consumption
          
        elif alg == 'kmeans_optimized':
          # kmeans_optimized stores a projected_matrix_samples + projected_matrix_clusters
          # 1. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
          # 2  projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
          
          annz_projected_matrix_samples = res['block_vector_data']['annz']
          # annz_projected_matrix_clusters was not measured (we use the annz_projected_matrix_samples as an approximation)
          annz_projected_matrix_clusters = annz_projected_matrix_samples
          
          projected_matrix_samples_mem_consumption = (annz_projected_matrix_samples * no_samples
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
                              
          projected_matrix_clusters_mem_consumption = (annz_projected_matrix_clusters * no_clusters_remaining
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
          
          mem_consumption = projected_matrix_samples_mem_consumption \
                            + projected_matrix_clusters_mem_consumption
        elif alg == 'yinyang':
          # yinyang stores a dense matrix to keep a lower bound to every of the t groups
          t = no_clusters_remaining / 10
          mem_consumption = no_samples * t * size_of_data_storage_element
          
        elif alg == 'fast_yinyang':
          # yinyang stores a dense matrix to keep a lower bound to every of the t groups + block vector projected matrices samples/clusters
          # 1. lower_bound_group_matrix = no_samples * t
          # 2. projected_matrix_samples = no_samples * dim ( = no_orthonormal_vectors)
          # 3. projected_matrix_clusters = no_clusters_remaining * dim ( = no_orthonormal_vectors)
          t = no_clusters_remaining / 10
          lower_bound_group_matrix_mem_consumption = no_samples * t * size_of_data_storage_element
          
          annz_projected_matrix_samples = res['block_vector_data']['annz']
          # annz_projected_matrix_clusters was not measured (we use the annz_projected_matrix_samples as an approximation)
          annz_projected_matrix_clusters = annz_projected_matrix_samples
          
          projected_matrix_samples_mem_consumption = (annz_projected_matrix_samples * no_samples
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
                              
          projected_matrix_clusters_mem_consumption = (annz_projected_matrix_clusters * no_clusters_remaining
                              * (size_of_data_storage_element + size_of_key_storage_element)) \
                              + ((no_samples + 1) * size_of_pointer_storage_element)
          
          mem_consumption = lower_bound_group_matrix_mem_consumption \
                            + projected_matrix_samples_mem_consumption \
                            + projected_matrix_clusters_mem_consumption
        else:
          raise Exception("please provide details for the memory consumption of %s" % alg)
          
        kmeans_duration_this_run = duration_kmeans
        
        if 'truncated_svd' in res:
          kmeans_duration_this_run += res['truncated_svd']['duration']
        
        
        
        mem_consumption = (mem_consumption / 1024.0) / 1024.0
        result_data[ds]['results'][no_clusters_remaining][alg]['duration'][run] = (float(mem_consumption), kmeans_duration_this_run)
        
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment to evaluate the structure of given datasets'
                                               ' compared to their precalculated counterparts')
  parser.add_argument('--dataset_folder', type=str, default="datasets", help='Path datasets are downloaded to.')
  parser.add_argument('--output_path', type=str, default="output_path", help='Path for the results.')
  parser.add_argument('--output_path_csv', type=str, default="output_path_csv", help='Path for the results as csv')
  parser.add_argument('--only_result_evaluation', dest='only_evaluation', action='store_true')
  parser.add_argument('--testmode', dest='testmode', action='store_true')
  parser.set_defaults(only_evaluation=False)
  parser.set_defaults(testmode=False)
  
  args = parser.parse_args()
  
  out_folder = abspath(args.output_path)
  out_folder_csv = abspath(args.output_path_csv)
  
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

  if not os.path.isdir(out_folder_csv):
    os.makedirs(out_folder_csv)

  result_evaluation(out_folder, out_folder_csv)

  #dataset_path = 
  #print(dataset_path)
