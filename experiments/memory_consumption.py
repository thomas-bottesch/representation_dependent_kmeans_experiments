from __future__ import print_function
from latex_plots_memory_consumption import create_plot
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

def result_evaluation_memory_consumption(out_folder, out_folder_csv):
  
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