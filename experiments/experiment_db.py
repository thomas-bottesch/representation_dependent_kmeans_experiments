import os
import cPickle
import json
from os.path import abspath, join, dirname

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
